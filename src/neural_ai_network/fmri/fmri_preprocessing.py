# fmri_preprocessing.py
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import matplotlib.pyplot as plt
from nilearn import image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel
import tempfile
import shutil
from pathlib import Path

@dataclass
class PreprocessingConfig:
    """Configuration for fMRI preprocessing pipeline."""
    # Input/Output
    save_interim: bool = True
    interim_dir: str = "./data/interim/fmri"
    
    # Slice Timing
    slice_timing: bool = True
    slice_order: str = "ascending"  # "ascending", "descending", "interleaved"
    slice_tr: Optional[float] = None  # If None, will use header info
    slice_ta: Optional[float] = None  # Acquisition time, if None will calculate from TR
    slice_ref_slice: int = 0  # Reference slice
    
    # Motion Correction
    motion_correction: bool = True
    moco_reference_vol: int = 0  # Reference volume for motion correction
    moco_cost_function: str = "mutualinfo"  # "mutualinfo", "corratio", "normcorr", "normmi", "leastsq"
    
    # Spatial Normalization
    normalization: bool = True
    normalization_template: str = "MNI152"  # "MNI152", "fsaverage"
    normalization_voxel_size: Tuple[int, int, int] = (2, 2, 2)  # Target voxel size
    
    # Registration
    coregistration: bool = True
    coreg_dof: int = 12  # Degrees of freedom for linear registration
    
    # Brain Extraction
    brain_extraction: bool = True
    bet_threshold: float = 0.5  # Fractional intensity threshold for BET
    
    # Spatial Smoothing
    spatial_smoothing: bool = True
    smoothing_fwhm: float = 6.0  # Full-width at half maximum in mm
    
    # Temporal Filtering
    temporal_filtering: bool = True
    highpass_cutoff: float = 0.01  # In Hz
    lowpass_cutoff: Optional[float] = None  # In Hz, None for no lowpass
    
    # Intensity Normalization
    intensity_normalization: bool = True
    intensity_norm_method: str = "global"  # "global", "rescale", "percentile"
    
    # Grand mean scaling
    grand_mean_scaling: bool = True
    grand_mean_value: int = 10000  # Target mean intensity value
    
    # Nuisance Regression
    nuisance_regression: bool = True
    nuisance_regressors: List[str] = field(default_factory=lambda: [
        "wm", "csf", "global", "motion"
    ])
    
    # Detrending
    detrending: bool = True
    detrending_order: int = 1  # 0 = mean removal, 1 = linear trend, etc.
    
    # Confound removal
    remove_confounds: bool = True
    n_compcor_components: int = 5  # CompCor components to extract
    
    # AROMA (Automatic Removal of Motion Artifacts)
    aroma: bool = False
    aroma_dim: int = 0  # 0 for automatic dimensionality estimation
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "PreprocessingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, file_path: str) -> str:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return file_path
    
    @classmethod
    def load(cls, file_path: str) -> "PreprocessingConfig":
        """Load config from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class FMRIPreprocessor:
    """
    Comprehensive fMRI preprocessing pipeline.
    
    This class implements a modular preprocessing pipeline for fMRI data,
    with configurable steps including slice timing correction, motion correction,
    spatial normalization, and confound estimation.
    """
    
    def __init__(self, config: Optional[Union[Dict, PreprocessingConfig]] = None):
        """
        Initialize the fMRI preprocessor.
        
        Args:
            config: Preprocessing configuration (dictionary or PreprocessingConfig)
        """
        self.logger = self._setup_logging()
        
        # Set configuration
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            self.config = PreprocessingConfig.from_dict(config)
        else:
            self.config = config
        
        # Create interim directory if needed
        if self.config.save_interim:
            os.makedirs(self.config.interim_dir, exist_ok=True)
        
        self.logger.info("fMRI Preprocessor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("FMRIPreprocessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def preprocess(self, func_img: nib.Nifti1Image, anat_img: Optional[nib.Nifti1Image] = None,
                 confounds: Optional[pd.DataFrame] = None, mask_img: Optional[nib.Nifti1Image] = None,
                 subject_id: Optional[str] = None) -> Dict:
        """
        Apply the complete preprocessing pipeline to fMRI data.
        
        Args:
            func_img: Functional MRI data (4D NIfTI)
            anat_img: Optional anatomical image (3D NIfTI)
            confounds: Optional confounds DataFrame
            mask_img: Optional brain mask
            subject_id: Optional subject identifier
            
        Returns:
            Dictionary with preprocessing results
        """
        if subject_id is None:
            subject_id = f"sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting preprocessing pipeline for {subject_id}")
        
        # Create result dictionary
        result = {
            "subject_id": subject_id,
            "inputs": {
                "functional": {
                    "shape": func_img.shape,
                    "voxel_dims": func_img.header.get_zooms(),
                    "n_volumes": func_img.shape[3] if len(func_img.shape) > 3 else 1
                }
            },
            "processing_steps": [],
            "outputs": {}
        }
        
        # Add anatomical info if provided
        if anat_img is not None:
            result["inputs"]["anatomical"] = {
                "shape": anat_img.shape,
                "voxel_dims": anat_img.header.get_zooms()
            }
        
        # Extract TR from header
        tr = func_img.header.get_zooms()[3] if len(func_img.header.get_zooms()) > 3 else None
        if tr is None:
            self.logger.warning("TR not found in NIfTI header, using default of 2.0s")
            tr = 2.0
        
        result["inputs"]["functional"]["tr"] = tr
        
        # Initialize intermediate images
        preprocessed_func = func_img
        preprocessed_anat = anat_img
        
        # Save raw input if requested
        if self.config.save_interim:
            self._save_interim(func_img, subject_id, "00_raw_func")
            if anat_img is not None:
                self._save_interim(anat_img, subject_id, "00_raw_anat")
        
        # 1. Motion Correction
        if self.config.motion_correction:
            preprocessed_func, motion_params, moco_info = self._apply_motion_correction(preprocessed_func, tr)
            result["processing_steps"].append({
                "step": "motion_correction",
                "info": moco_info
            })
            result["outputs"]["motion_parameters"] = motion_params
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "01_motion_corrected")
        
        # 2. Slice Timing Correction
        if self.config.slice_timing:
            preprocessed_func, stc_info = self._apply_slice_timing_correction(preprocessed_func, tr)
            result["processing_steps"].append({
                "step": "slice_timing_correction",
                "info": stc_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "02_slice_time_corrected")
        
        # 3. Brain Extraction (for anatomical if provided)
        if anat_img is not None and self.config.brain_extraction:
            preprocessed_anat, brain_mask, bet_info = self._apply_brain_extraction(preprocessed_anat)
            result["processing_steps"].append({
                "step": "brain_extraction",
                "info": bet_info
            })
            result["outputs"]["brain_mask"] = brain_mask
            
            if self.config.save_interim:
                self._save_interim(preprocessed_anat, subject_id, "03_brain_extracted")
                self._save_interim(brain_mask, subject_id, "03_brain_mask")
        
        # 4. Coregistration (anatomical to functional)
        if anat_img is not None and self.config.coregistration:
            preprocessed_anat, coreg_info = self._apply_coregistration(preprocessed_func, preprocessed_anat)
            result["processing_steps"].append({
                "step": "coregistration",
                "info": coreg_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_anat, subject_id, "04_coregistered")
        
        # 5. Spatial Normalization
        if self.config.normalization:
            preprocessed_func, norm_transform, norm_info = self._apply_normalization(
                preprocessed_func, 
                preprocessed_anat if anat_img is not None else None
            )
            result["processing_steps"].append({
                "step": "spatial_normalization",
                "info": norm_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "05_normalized")
        
        # 6. Spatial Smoothing
        if self.config.spatial_smoothing:
            preprocessed_func, smooth_info = self._apply_spatial_smoothing(preprocessed_func)
            result["processing_steps"].append({
                "step": "spatial_smoothing",
                "info": smooth_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "06_smoothed")
        
        # 7. Temporal Filtering
        if self.config.temporal_filtering:
            preprocessed_func, filter_info = self._apply_temporal_filtering(preprocessed_func, tr)
            result["processing_steps"].append({
                "step": "temporal_filtering",
                "info": filter_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "07_filtered")
        
        # 8. Intensity Normalization
        if self.config.intensity_normalization:
            preprocessed_func, inorm_info = self._apply_intensity_normalization(preprocessed_func)
            result["processing_steps"].append({
                "step": "intensity_normalization",
                "info": inorm_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "08_intensity_normalized")
        
        # 9. Confound Estimation and Regression
        if confounds is not None and self.config.remove_confounds:
            preprocessed_func, confound_info = self._apply_confound_regression(
                preprocessed_func, 
                confounds,
                result.get("outputs", {}).get("motion_parameters", None)
            )
            result["processing_steps"].append({
                "step": "confound_regression",
                "info": confound_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "09_confound_regressed")
        
        # 10. Grand Mean Scaling
        if self.config.grand_mean_scaling:
            preprocessed_func, gms_info = self._apply_grand_mean_scaling(preprocessed_func)
            result["processing_steps"].append({
                "step": "grand_mean_scaling",
                "info": gms_info
            })
            
            if self.config.save_interim:
                self._save_interim(preprocessed_func, subject_id, "10_grand_mean_scaled")
        
        # 11. AROMA (if enabled)
        if self.config.aroma:
            try:
                preprocessed_func, aroma_info = self._apply_aroma(
                    preprocessed_func, 
                    result.get("outputs", {}).get("motion_parameters", None),
                    mask_img
                )
                result["processing_steps"].append({
                    "step": "aroma",
                    "info": aroma_info
                })
                
                if self.config.save_interim:
                    self._save_interim(preprocessed_func, subject_id, "11_aroma")
            except Exception as e:
                self.logger.error(f"Error in AROMA processing: {e}")
                self.logger.warning("Continuing without AROMA")
        
        # Save final preprocessed images
        final_func_path = os.path.join(self.config.interim_dir, f"{subject_id}_preprocessed_func.nii.gz")
        nib.save(preprocessed_func, final_func_path)
        
        final_anat_path = None
        if anat_img is not None:
            final_anat_path = os.path.join(self.config.interim_dir, f"{subject_id}_preprocessed_anat.nii.gz")
            nib.save(preprocessed_anat, final_anat_path)
        
        # Add outputs to result
        result["outputs"]["preprocessed_functional"] = {
            "path": final_func_path,
            "shape": preprocessed_func.shape,
            "voxel_dims": preprocessed_func.header.get_zooms()
        }
        
        if final_anat_path:
            result["outputs"]["preprocessed_anatomical"] = {
                "path": final_anat_path,
                "shape": preprocessed_anat.shape,
                "voxel_dims": preprocessed_anat.header.get_zooms()
            }
        
        # Generate quality control report
        try:
            qc_report_path = self._generate_qc_report(
                original_func=func_img,
                preprocessed_func=preprocessed_func,
                original_anat=anat_img,
                preprocessed_anat=preprocessed_anat,
                motion_params=result.get("outputs", {}).get("motion_parameters", None),
                subject_id=subject_id
            )
            result["outputs"]["qc_report"] = qc_report_path
        except Exception as e:
            self.logger.error(f"Error generating QC report: {e}")
        
        self.logger.info(f"Preprocessing completed for {subject_id}")
        
        # Store final preprocessed images
        result["preprocessed_functional"] = preprocessed_func
        if anat_img is not None:
            result["preprocessed_anatomical"] = preprocessed_anat
        
        return result
    
    def _apply_motion_correction(self, func_img: nib.Nifti1Image, tr: float) -> Tuple[nib.Nifti1Image, np.ndarray, Dict]:
        """Apply motion correction to functional image."""
        self.logger.info("Applying motion correction")
        
        try:
            from nilearn.image import index_img
            from nipype.interfaces import fsl
            from nipype.interfaces.fsl import MCFLIRT
            import tempfile
            import shutil
            
            # Create temporary directory for MCFLIRT
            temp_dir = tempfile.mkdtemp()
            
            # Save input image to temp dir
            input_file = os.path.join(temp_dir, "input.nii.gz")
            nib.save(func_img, input_file)
            
            # Run MCFLIRT
            mcflirt = MCFLIRT()
            mcflirt.inputs.in_file = input_file
            mcflirt.inputs.cost = self.config.moco_cost_function
            mcflirt.inputs.mean_vol = True
            mcflirt.inputs.output_type = "NIFTI_GZ"
            mcflirt.inputs.ref_vol = self.config.moco_reference_vol
            mcflirt.inputs.out_file = os.path.join(temp_dir, "mcf.nii.gz")
            mcflirt.inputs.save_plots = True
            mcflirt.inputs.save_rms = True
            mcflirt.run()
            
            # Load motion-corrected image
            mc_img = nib.load(os.path.join(temp_dir, "mcf.nii.gz"))
            
            # Load motion parameters
            motion_params_file = os.path.join(temp_dir, "mcf.par")
            motion_params = np.loadtxt(motion_params_file)
            
            # Calculate motion statistics
            mean_displacement = np.mean(np.abs(motion_params), axis=0)
            max_displacement = np.max(np.abs(motion_params), axis=0)
            
            # Calculate framewise displacement
            fd = np.zeros(motion_params.shape[0])
            
            # For the first timepoint, FD is unknown
            fd[0] = 0
            
            # For subsequent timepoints, calculate FD
            for i in range(1, motion_params.shape[0]):
                # Calculate absolute displacement between consecutive timepoints
                delta = np.abs(motion_params[i] - motion_params[i-1])
                
                # Convert rotations to mm displacement (assuming brain radius of 50mm)
                delta[3:] = delta[3:] * 50
                
                # FD is the sum of the absolute displacements
                fd[i] = np.sum(delta)
            
            # Calculate motion metrics
            mean_fd = np.mean(fd)
            max_fd = np.max(fd)
            num_fd_above_threshold = np.sum(fd > 0.5)  # 0.5mm is a common threshold
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Return motion-corrected image, motion parameters, and info
            return mc_img, motion_params, {
                "mean_displacement": mean_displacement.tolist(),
                "max_displacement": max_displacement.tolist(),
                "mean_framewise_displacement": float(mean_fd),
                "max_framewise_displacement": float(max_fd),
                "volumes_with_fd_above_threshold": int(num_fd_above_threshold),
                "reference_volume": self.config.moco_reference_vol,
                "cost_function": self.config.moco_cost_function
            }
        except Exception as e:
            self.logger.error(f"Error in motion correction: {e}")
            raise
    
    def _apply_slice_timing_correction(self, func_img: nib.Nifti1Image, tr: float) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply slice timing correction to functional image."""
        self.logger.info("Applying slice timing correction")
        
        try:
            from nilearn.image import slice_time_correction
            
            # Get number of slices from the third dimension
            n_slices = func_img.shape[2]
            
            # Set slice order based on configuration
            if self.config.slice_order == "ascending":
                slice_times = np.arange(n_slices) / n_slices * tr
            elif self.config.slice_order == "descending":
                slice_times = np.arange(n_slices)[::-1] / n_slices * tr
            elif self.config.slice_order == "interleaved":
                # Common interleaved acquisition (0, 2, 4, ..., 1, 3, 5, ...)
                slice_times = np.zeros(n_slices)
                slice_times[::2] = np.arange(n_slices // 2) / n_slices * tr
                slice_times[1::2] = np.arange(n_slices // 2, n_slices) / n_slices * tr
            else:
                # Default to ascending
                self.logger.warning(f"Unknown slice order: {self.config.slice_order}, using ascending")
                slice_times = np.arange(n_slices) / n_slices * tr
            
            # Calculate acquisition time (TA) if not provided
            if self.config.slice_ta is None:
                # TA = TR - (TR / n_slices)
                ta = tr - (tr / n_slices)
            else:
                ta = self.config.slice_ta
            
            # Apply slice timing correction
            stc_img = slice_time_correction(
                func_img,
                slice_times,
                ref_slice=self.config.slice_ref_slice,
                t_r=tr
            )
            
            return stc_img, {
                "slice_order": self.config.slice_order,
                "n_slices": int(n_slices),
                "tr": float(tr),
                "ta": float(ta),
                "reference_slice": int(self.config.slice_ref_slice)
            }
        except Exception as e:
            self.logger.error(f"Error in slice timing correction: {e}")
            raise
    
    def _apply_brain_extraction(self, anat_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, Dict]:
        """Apply brain extraction to anatomical image."""
        self.logger.info("Applying brain extraction")
        
        try:
            from nilearn.masking import compute_brain_mask
            
            # Compute brain mask
            mask_img = compute_brain_mask(anat_img, threshold=self.config.bet_threshold)
            
            # Apply mask to anatomical image
            brain_img = image.math_img("img1 * img2", img1=anat_img, img2=mask_img)
            
            # Calculate statistics
            mask_data = mask_img.get_fdata()
            total_voxels = mask_data.size
            brain_voxels = np.sum(mask_data > 0)
            brain_ratio = brain_voxels / total_voxels
            
            return brain_img, mask_img, {
                "threshold": float(self.config.bet_threshold),
                "brain_voxels": int(brain_voxels),
                "total_voxels": int(total_voxels),
                "brain_ratio": float(brain_ratio)
            }
        except Exception as e:
            self.logger.error(f"Error in brain extraction: {e}")
            raise
    
    def _apply_coregistration(self, func_img: nib.Nifti1Image, anat_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict]:
        """Coregister anatomical to functional image."""
        self.logger.info("Applying coregistration")
        
        try:
            from nilearn.image import index_img, mean_img, resample_to_img
            
            # Extract reference volume from functional image
            if func_img.ndim == 4:
                ref_func = mean_img(func_img)  # Use mean functional as reference
            else:
                ref_func = func_img
            
            # Perform coregistration
            coreg_anat = resample_to_img(
                anat_img,
                ref_func,
                interpolation='continuous'
            )
            
            return coreg_anat, {
                "degrees_of_freedom": int(self.config.coreg_dof),
                "interpolation": "continuous",
                "reference": "mean_functional_volume"
            }
        except Exception as e:
            self.logger.error(f"Error in coregistration: {e}")
            raise
    
    def _apply_normalization(self, func_img: nib.Nifti1Image, 
                           anat_img: Optional[nib.Nifti1Image] = None) -> Tuple[nib.Nifti1Image, Any, Dict]:
        """Apply spatial normalization to functional image."""
        self.logger.info(f"Applying spatial normalization to {self.config.normalization_template} template")
        
        try:
            from nilearn.image import resample_to_img, resample_img
            
            # Get template image
            if self.config.normalization_template == "MNI152":
                from nilearn.datasets import load_mni152_template
                template = load_mni152_template()
            elif self.config.normalization_template == "fsaverage":
                from nilearn.datasets import fetch_surf_fsaverage
                fsaverage = fetch_surf_fsaverage()
                template = fsaverage['T1w']
            else:
                # Use custom template if specified
                template_path = self.config.normalization_template
                if os.path.exists(template_path):
                    template = nib.load(template_path)
                else:
                    raise ValueError(f"Unknown template: {self.config.normalization_template}")
            
            # Create target affine with desired voxel size
            voxel_size = self.config.normalization_voxel_size
            target_affine = np.diag(list(voxel_size) + [1])
            
            # Normalize anatomical to template if provided
            anat_transformed = None
            if anat_img is not None:
                anat_transformed = resample_img(
                    anat_img,
                    target_affine=target_affine,
                    target_shape=template.shape[:3],
                    interpolation="continuous"
                )
            
            # Apply normalization to functional data
            func_transformed = resample_img(
                func_img,
                target_affine=target_affine,
                target_shape=template.shape[:3],
                interpolation="continuous"
            )
            
            return func_transformed, target_affine, {
                "template": self.config.normalization_template,
                "voxel_size": list(voxel_size),
                "target_shape": template.shape[:3],
                "interpolation": "continuous"
            }
        except Exception as e:
            self.logger.error(f"Error in spatial normalization: {e}")
            raise
    
    def _apply_spatial_smoothing(self, func_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply spatial smoothing to functional image."""
        self.logger.info(f"Applying spatial smoothing with FWHM={self.config.smoothing_fwhm}mm")
        
        try:
            from nilearn.image import smooth_img
            
            # Apply smoothing
            fwhm = self.config.smoothing_fwhm
            smoothed_img = smooth_img(func_img, fwhm=fwhm)
            
            return smoothed_img, {
                "fwhm_mm": float(fwhm)
            }
        except Exception as e:
            self.logger.error(f"Error in spatial smoothing: {e}")
            raise
    
    def _apply_temporal_filtering(self, func_img: nib.Nifti1Image, tr: float) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply temporal filtering to functional image."""
        self.logger.info(f"Applying temporal filtering (high-pass={self.config.highpass_cutoff}Hz, low-pass={self.config.lowpass_cutoff}Hz)")
        
        try:
            from nilearn.image import clean_img
            
            # Apply temporal filtering
            filtered_img = clean_img(
                func_img,
                high_pass=self.config.highpass_cutoff,
                low_pass=self.config.lowpass_cutoff,
                t_r=tr
            )
            
            return filtered_img, {
                "highpass_cutoff_hz": float(self.config.highpass_cutoff),
                "lowpass_cutoff_hz": float(self.config.lowpass_cutoff) if self.config.lowpass_cutoff else None,
                "tr": float(tr)
            }
        except Exception as e:
            self.logger.error(f"Error in temporal filtering: {e}")
            raise
    
    def _apply_intensity_normalization(self, func_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply intensity normalization to functional image."""
        self.logger.info(f"Applying intensity normalization using method: {self.config.intensity_norm_method}")
        
        try:
            # Get data array
            data = func_img.get_fdata()
            
            # Apply normalization based on method
            if self.config.intensity_norm_method == "global":
                # Global mean normalization
                global_mean = np.mean(data)
                normalized_data = data / global_mean * 100  # Scale to percent of global mean
                
                norm_info = {
                    "method": "global",
                    "global_mean": float(global_mean),
                    "scale_factor": 100
                }
            
            elif self.config.intensity_norm_method == "rescale":
                # Min-max rescaling to [0, 1]
                min_val = np.min(data)
                max_val = np.max(data)
                normalized_data = (data - min_val) / (max_val - min_val)
                
                norm_info = {
                    "method": "rescale",
                    "original_min": float(min_val),
                    "original_max": float(max_val),
                    "target_range": [0, 1]
                }
            
            elif self.config.intensity_norm_method == "percentile":
                # Percentile-based normalization
                p01 = np.percentile(data, 1)
                p99 = np.percentile(data, 99)
                normalized_data = np.clip(data, p01, p99)  # Clip to 1-99 percentile
                normalized_data = (normalized_data - p01) / (p99 - p01)  # Rescale to [0, 1]
                
                norm_info = {
                    "method": "percentile",
                    "percentile_min": float(p01),
                    "percentile_max": float(p99),
                    "percentile_range": [1, 99],
                    "target_range": [0, 1]
                }
            
            else:
                self.logger.warning(f"Unknown normalization method: {self.config.intensity_norm_method}")
                return func_img, {"method": "none", "reason": "unknown_method"}
            
            # Create new image with normalized data
            normalized_img = nib.Nifti1Image(normalized_data, func_img.affine, func_img.header)
            
            return normalized_img, norm_info
        except Exception as e:
            self.logger.error(f"Error in intensity normalization: {e}")
            raise
    
    def _apply_confound_regression(self, func_img: nib.Nifti1Image, 
                                 confounds: pd.DataFrame,
                                 motion_params: Optional[np.ndarray] = None) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply confound regression to functional image."""
        self.logger.info("Applying confound regression")
        
        try:
            from nilearn.image import clean_img
            
            # Prepare confound regressors
            selected_confounds = []
            confound_info = {"included_regressors": []}
            
            # Include specified confound types
            if "wm" in self.config.nuisance_regressors and "white_matter" in confounds.columns:
                selected_confounds.append(confounds["white_matter"])
                confound_info["included_regressors"].append("white_matter")
            
            if "csf" in self.config.nuisance_regressors and "csf" in confounds.columns:
                selected_confounds.append(confounds["csf"])
                confound_info["included_regressors"].append("csf")
            
            if "global" in self.config.nuisance_regressors and "global_signal" in confounds.columns:
                selected_confounds.append(confounds["global_signal"])
                confound_info["included_regressors"].append("global_signal")
            
            # Include motion parameters if available
            if "motion" in self.config.nuisance_regressors:
                if motion_params is not None:
                    # Use motion parameters from motion correction
                    selected_confounds.append(pd.DataFrame(motion_params))
                    confound_info["included_regressors"].append("motion_parameters")
                elif any(col.startswith(("trans_", "rot_")) for col in confounds.columns):
                    # Use motion parameters from confounds DataFrame
                    motion_cols = [col for col in confounds.columns if col.startswith(("trans_", "rot_"))]
                    selected_confounds.append(confounds[motion_cols])
                    confound_info["included_regressors"].extend(motion_cols)
            
            # Prepare combined confounds
            if selected_confounds:
                combined_confounds = pd.concat(selected_confounds, axis=1)
                combined_confounds = combined_confounds.fillna(0)  # Fill any NaN values
                
                # Apply confound regression
                denoised_img = clean_img(
                    func_img,
                    confounds=combined_confounds,
                    detrend=self.config.detrending,
                    standardize=False  # Already done in intensity normalization
                )
                
                confound_info["n_confound_regressors"] = combined_confounds.shape[1]
                confound_info["detrending"] = self.config.detrending
                
                return denoised_img, confound_info
            else:
                self.logger.warning("No confound regressors found or selected")
                return func_img, {"included_regressors": [], "reason": "no_regressors_found"}
        except Exception as e:
            self.logger.error(f"Error in confound regression: {e}")
            raise
    
    def _apply_grand_mean_scaling(self, func_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply grand mean scaling to functional image."""
        self.logger.info(f"Applying grand mean scaling to target value: {self.config.grand_mean_value}")
        
        try:
            # Get data array
            data = func_img.get_fdata()
            
            # Calculate grand mean
            grand_mean = np.mean(data)
            
            # Scale to target value
            scaled_data = data * (self.config.grand_mean_value / grand_mean)
            
            # Create new image
            scaled_img = nib.Nifti1Image(scaled_data, func_img.affine, func_img.header)
            
            return scaled_img, {
                "original_grand_mean": float(grand_mean),
                "target_value": int(self.config.grand_mean_value),
                "scaling_factor": float(self.config.grand_mean_value / grand_mean)
            }
        except Exception as e:
            self.logger.error(f"Error in grand mean scaling: {e}")
            raise
    
    def _apply_aroma(self, func_img: nib.Nifti1Image, 
                    motion_params: Optional[np.ndarray] = None, 
                    mask_img: Optional[nib.Nifti1Image] = None) -> Tuple[nib.Nifti1Image, Dict]:
        """Apply ICA-AROMA for motion artifact removal."""
        self.logger.info("Applying ICA-AROMA for motion artifact removal")
        
        try:
            # For true ICA-AROMA, we would need the external tool installed
            # Here we'll implement a simplified version using nilearn's ICA
            from nilearn.decomposition import CanICA
            from nilearn.image import index_img, mean_img
            from nilearn.masking import compute_epi_mask
            
            # Create mask if not provided
            if mask_img is None:
                mask_img = compute_epi_mask(func_img)
            
            # Apply ICA
            n_components = self.config.aroma_dim if self.config.aroma_dim > 0 else 20
            
            ica = CanICA(
                n_components=n_components,
                mask=mask_img,
                random_state=42,
                memory='nilearn_cache',
                memory_level=2,
                verbose=0
            )
            
            ica.fit(func_img)
            components_img = ica.components_img_
            components = ica.components_
            
            # Identify motion-related components
            motion_related = []
            
            # We need motion parameters to identify motion-related components
            if motion_params is not None:
                # Calculate correlation between each component and motion parameters
                for i in range(components.shape[0]):
                    component_ts = ica.transform(func_img)[:, i]
                    
                    # Calculate correlation with each motion parameter
                    correlations = []
                    for j in range(motion_params.shape[1]):
                        corr = np.corrcoef(component_ts, motion_params[:, j])[0, 1]
                        correlations.append(abs(corr))  # Use absolute correlation
                    
                    # If component is highly correlated with any motion parameter, mark it
                    if max(correlations) > 0.5:  # Arbitrary threshold
                        motion_related.append(i)
            
            # If we couldn't identify components based on motion parameters,
            # use edge fraction and spatial metrics (simplified AROMA approach)
            if not motion_related:
                # Calculate edge fraction for each component
                edge_fractions = []
                
                for i in range(components.shape[0]):
                    comp_img = index_img(components_img, i)
                    comp_data = comp_img.get_fdata()
                    
                    # Calculate edge fraction (voxels on edge / total voxels)
                    mask = comp_data > np.percentile(comp_data, 95)  # Threshold component map
                    total_voxels = np.sum(mask)
                    
                    if total_voxels > 0:
                        # Define edges as voxels on the boundary of the volume
                        edges = np.zeros_like(mask)
                        edges[0, :, :] = edges[-1, :, :] = edges[:, 0, :] = edges[:, -1, :] = edges[:, :, 0] = edges[:, :, -1] = 1
                        edge_voxels = np.sum(mask & edges)
                        edge_fraction = edge_voxels / total_voxels
                    else:
                        edge_fraction = 0
                    
                    edge_fractions.append(edge_fraction)
                
                # Components with high edge fraction are likely motion-related
                for i, ef in enumerate(edge_fractions):
                    if ef > 0.2:  # Arbitrary threshold
                        motion_related.append(i)
            
            # Remove motion-related components
            if motion_related:
                # Create a denoised version of the functional image
                from nilearn.image import clean_img
                
                # Get component time series
                component_ts = ica.transform(func_img)
                
                # Create regressor matrix for motion components
                motion_regressors = component_ts[:, motion_related]
                
                # Regress out motion components
                denoised_img = clean_img(
                    func_img,
                    confounds=motion_regressors,
                    mask_img=mask_img,
                    standardize=False  # Don't standardize again
                )
                
                aroma_info = {
                    "n_components": int(n_components),
                    "motion_related_components": [int(i) for i in motion_related],
                    "n_motion_related": len(motion_related)
                }
                
                return denoised_img, aroma_info
            else:
                self.logger.warning("No motion-related components identified")
                return func_img, {
                    "n_components": int(n_components),
                    "motion_related_components": [],
                    "n_motion_related": 0,
                    "warning": "no_motion_components_identified"
                }
        except Exception as e:
            self.logger.error(f"Error in AROMA processing: {e}")
            raise
    
    def _generate_qc_report(self, original_func: nib.Nifti1Image, 
                          preprocessed_func: nib.Nifti1Image,
                          original_anat: Optional[nib.Nifti1Image] = None,
                          preprocessed_anat: Optional[nib.Nifti1Image] = None,
                          motion_params: Optional[np.ndarray] = None,
                          subject_id: str = "subject") -> str:
        """Generate quality control report with visualizations."""
        self.logger.info("Generating QC report")
        
        # Create QC directory
        qc_dir = os.path.join(self.config.interim_dir, f"{subject_id}_qc")
        os.makedirs(qc_dir, exist_ok=True)
        
        # 1. Plot mean functional image before and after preprocessing
        mean_orig = image.mean_img(original_func)
        mean_prep = image.mean_img(preprocessed_func)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plotting.plot_epi(mean_orig, display_mode='z', title="Original Mean", axes=plt.gca())
        plt.subplot(1, 2, 2)
        plotting.plot_epi(mean_prep, display_mode='z', title="Preprocessed Mean", axes=plt.gca())
        plt.savefig(os.path.join(qc_dir, "mean_comparison.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Plot functional to anatomical alignment (if anatomical provided)
        if original_anat is not None and preprocessed_anat is not None:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            display = plotting.plot_anat(original_anat, display_mode='z', title="Original Anatomical", axes=plt.gca())
            plt.subplot(1, 2, 2)
            display = plotting.plot_anat(preprocessed_anat, display_mode='z', title="Preprocessed Anatomical", axes=plt.gca())
            plt.savefig(os.path.join(qc_dir, "anat_comparison.png"), dpi=100, bbox_inches='tight')
            plt.close()
            
            # Overlay functional on anatomical
            plt.figure(figsize=(10, 5))
            display = plotting.plot_epi(mean_prep, display_mode='z', title="Func/Anat Overlay", axes=plt.gca())
            if display._add_overlay is not None:  # Check if method exists
                display._add_overlay(preprocessed_anat, cmap=plt.cm.Reds, alpha=0.5)
            plt.savefig(os.path.join(qc_dir, "func_anat_overlay.png"), dpi=100, bbox_inches='tight')
            plt.close()
        
        # 3. Plot motion parameters if available
        if motion_params is not None:
            plt.figure(figsize=(10, 6))
            n_vols = motion_params.shape[0]
            
            # Plot translations
            plt.subplot(2, 1, 1)
            for i in range(3):  # x, y, z translations
                plt.plot(np.arange(n_vols), motion_params[:, i], label=f"{'XYZ'[i]}-trans")
            plt.xlabel("Volume")
            plt.ylabel("Translation (mm)")
            plt.legend()
            plt.title("Head Motion: Translations")
            
            # Plot rotations
            plt.subplot(2, 1, 2)
            for i in range(3, 6):  # pitch, roll, yaw rotations
                plt.plot(np.arange(n_vols), motion_params[:, i], label=f"{'XYZ'[i-3]}-rot")
            plt.xlabel("Volume")
            plt.ylabel("Rotation (radians)")
            plt.legend()
            plt.title("Head Motion: Rotations")
            
            plt.tight_layout()
            plt.savefig(os.path.join(qc_dir, "motion_parameters.png"), dpi=100, bbox_inches='tight')
            plt.close()
            
            # Calculate and plot framewise displacement
            fd = np.zeros(n_vols)
            
            # For the first timepoint, FD is unknown
            fd[0] = 0
            
            # For subsequent timepoints, calculate FD
            for i in range(1, n_vols):
                # Calculate absolute displacement between consecutive timepoints
                delta = np.abs(motion_params[i] - motion_params[i-1])
                
                # Convert rotations to mm displacement (assuming brain radius of 50mm)
                delta[3:] = delta[3:] * 50
                
                # FD is the sum of the absolute displacements
                fd[i] = np.sum(delta)
            
            # Plot framewise displacement
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(n_vols), fd, 'b-')
            plt.axhline(y=0.5, color='r', linestyle='--', label='0.5mm threshold')
            plt.xlabel("Volume")
            plt.ylabel("Framewise Displacement (mm)")
            plt.title("Framewise Displacement")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(qc_dir, "framewise_displacement.png"), dpi=100, bbox_inches='tight')
            plt.close()
        
        # 4. Plot temporal SNR
        from nilearn.signal import clean
        from nilearn.masking import compute_epi_mask, apply_mask
        
        # Create mask
        mask = compute_epi_mask(preprocessed_func)
        
        # Extract time series data
        data = apply_mask(preprocessed_func, mask)
        
        # Calculate temporal SNR (mean / std along time axis)
        mean_ts = np.mean(data, axis=0)
        std_ts = np.std(data, axis=0)
        tsnr = mean_ts / std_ts
        
        # Create tSNR map
        tsnr_map = masking.unmask(tsnr, mask)
        
        # Plot tSNR map
        plt.figure(figsize=(10, 5))
        plotting.plot_stat_map(tsnr_map, display_mode='z', title="Temporal SNR Map", threshold=1.0)
        plt.savefig(os.path.join(qc_dir, "tsnr_map.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        # 5. Create HTML report
        html_path = os.path.join(qc_dir, "qc_report.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
            <html>
            <head>
                <title>fMRI Preprocessing QC Report: {subject_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #444; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                    img {{ max-width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>fMRI Preprocessing QC Report: {subject_id}</h1>
                    
                    <div class="section">
                        <h2>Functional Images</h2>
                        <p>Comparison of original and preprocessed mean functional image:</p>
                        <img src="mean_comparison.png" alt="Mean Functional Comparison">
                    </div>
            """)
            
            # Add anatomical section if available
            if original_anat is not None:
                f.write(f"""
                    <div class="section">
                        <h2>Anatomical Images</h2>
                        <p>Comparison of original and preprocessed anatomical image:</p>
                        <img src="anat_comparison.png" alt="Anatomical Comparison">
                        
                        <p>Functional-Anatomical Overlay:</p>
                        <img src="func_anat_overlay.png" alt="Functional-Anatomical Overlay">
                    </div>
                """)
            
            # Add motion section if available
            if motion_params is not None:
                f.write(f"""
                    <div class="section">
                        <h2>Motion Parameters</h2>
                        <p>Head motion during scan:</p>
                        <img src="motion_parameters.png" alt="Motion Parameters">
                        
                        <p>Framewise Displacement:</p>
                        <img src="framewise_displacement.png" alt="Framewise Displacement">
                    </div>
                """)
            
            # Add tSNR section
            f.write(f"""
                    <div class="section">
                        <h2>Temporal SNR</h2>
                        <p>Temporal signal-to-noise ratio map:</p>
                        <img src="tsnr_map.png" alt="Temporal SNR Map">
                    </div>
                    
                </div>
            </body>
            </html>
            """)
        
        self.logger.info(f"QC report generated at {html_path}")
        return html_path
    
    def _save_interim(self, img: nib.Nifti1Image, subject_id: str, suffix: str) -> str:
        """Save intermediate processing step as NIfTI file."""
        if not self.config.save_interim:
            return None
        
        output_path = os.path.join(self.config.interim_dir, f"{subject_id}_{suffix}.nii.gz")
        nib.save(img, output_path)
        self.logger.info(f"Saved interim result: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize with default config
    preprocessor = FMRIPreprocessor()
    
    # Example: Process a sample fMRI dataset
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess fMRI data')
    parser.add_argument('--func', type=str, help='Path to functional NIfTI file')
    parser.add_argument('--anat', type=str, help='Path to anatomical NIfTI file (optional)')
    parser.add_argument('--confounds', type=str, help='Path to confounds TSV file (optional)')
    parser.add_argument('--subject', type=str, default='test_subject', help='Subject identifier')
    parser.add_argument('--config', type=str, help='Path to config JSON file (optional)')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        config = PreprocessingConfig.load(args.config)
        preprocessor = FMRIPreprocessor(config)
    
    # Check for required inputs
    if args.func:
        # Load functional data
        func_img = nib.load(args.func)
        
        # Load anatomical data if provided
        anat_img = None
        if args.anat:
            anat_img = nib.load(args.anat)
        
        # Load confounds if provided
        confounds = None
        if args.confounds:
            confounds = pd.read_csv(args.confounds, sep='\t')
        
        # Run preprocessing
        result = preprocessor.preprocess(
            func_img=func_img,
            anat_img=anat_img,
            confounds=confounds,
            subject_id=args.subject
        )
        
        print("\nPreprocessing completed:")
        print(f"- Steps performed: {', '.join([step['step'] for step in result['processing_steps']])}")
        print(f"- Output directory: {preprocessor.config.interim_dir}")
        if 'qc_report' in result.get('outputs', {}):
            print(f"- QC report: {result['outputs']['qc_report']}")
    else:
        print("Please provide a functional image (--func)")