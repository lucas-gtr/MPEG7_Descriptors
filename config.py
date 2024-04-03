from src.descriptors.dominant_color import DominantColorDescriptor
from src.descriptors.color_layout import ColorLayoutDescriptor
from src.descriptors.color_structure import ColorStructureDescriptor

# Variables for Dominant Color Descriptor
Td = 20  # Usually between 10 and 20
alpha = 1.5  # Usually between 1.0 and 1.5

# Variables for Color Layout Descriptor
y_coeff_number = 6
c_coeff_number = 3  # Usually there are more Y coeff number than Cr, Cb
# Weights for distance calculation
w_y = 1
w_cr = 1
w_cb = 1


# Variables for Color Structure Descriptor
n_quantization = 64

DESCRIPTOR_LIST = {
    "DCD": DominantColorDescriptor(Td, alpha),
    "CLD": ColorLayoutDescriptor(y_coeff_number, c_coeff_number, w_y, w_cr, w_cb),
    "CSD": ColorStructureDescriptor(n_quantization)
}
