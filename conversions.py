# CONVERSIONS

# lengths
um_2_m = 1E-6
mm_2_m = 1E-3
m_2_mm = 1E3
m_2_um = 1E6

# volumes
uL_2_mL = 1E-3

# times
s_2_ms = 1E3
min_2_s = 60
s_2_min = 1/60

# pressures
Pa_2_bar = 1E-5

# flow rates
uLmin_2_m3s = 1/60E9
m3s_2_uLmin = 60E9

# pixel conversions for microscope objectives
pix_per_um_dict = {4 : 1.34, # measured based on reference dot for 4x objective
                10 : 3.54} # see get_pixel_width_from_calibration_slide.py
