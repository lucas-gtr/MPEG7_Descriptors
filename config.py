from src.descriptors.dominant_color import DominantColorDescriptor
from src.descriptors.color_layout import ColorLayoutDescriptor
from src.descriptors.color_structure import ColorStructureDescriptor

# Variables for Dominant Color Descriptor
Td = 20  # For distance calculation (usually between 10 and 20)
alpha = 1.5  # For distance calculation (usually between 1.0 and 1.5)

# Variables for Color Layout Descriptor
y_coeff_number = 6  # For descriptor size
c_coeff_number = 3  # For descriptor size (usually there are more Y coeff number than Cr, Cb)
# Weights for distance calculation
w_y = 1
w_cr = 1
w_cb = 1


# Variables for Color Structure Descriptor
n_quantization = 64  # For descriptor size
assert n_quantization in {32, 64, 128, 256}, "The value of n_quantization must be 32, 64, 128, or 256."


DESCRIPTOR_LIST = {
    "DCD": DominantColorDescriptor(Td, alpha),
    "CLD": ColorLayoutDescriptor(y_coeff_number, c_coeff_number, w_y, w_cr, w_cb),
    "CSD": ColorStructureDescriptor(n_quantization)
}
