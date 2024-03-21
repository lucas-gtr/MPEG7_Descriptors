class Descriptor:
    """
    Base class for image descriptors.

    Methods:
        __init__: Initializes the Descriptor class.
        get_descriptor: Method to compute the descriptor for an image.
        get_distance: Method to compute the distance between two descriptors.
    """
    def __init__(self):
        super().__init__()

    def get_descriptor(self, img) -> str:
        """
        Compute the descriptor for an image.

        Parameters:
            img (numpy.ndarray): The input image.

        Returns:
            str: The computed descriptor for the input image.
        """
        pass

    def get_distance(self, d_1, d_2) -> float:
        """
        Compute the distance between two descriptors.

        Parameters:
            d_1 (numpy.ndarray): Descriptor array of the first image.
            d_2 (numpy.ndarray): Descriptor array of the second image.

        Returns:
            float: The distance between the two descriptors.
        """
        pass
