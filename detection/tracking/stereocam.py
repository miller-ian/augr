import abc

class StereoCam(abc.ABC):
    """
    The Abstract Base Class for Stereoscopic Cameras.
    
    Provides a functional interface to receive data from external cameras.
    """
    @abc.abstractmethod
    def get_pixels(self):
        """Yields a 3xHEIGHTxWIDTH list of floats representing the RGB pixel values of the current feed."""
        pass

    @abc.abstractmethod
    def get_depth(self):
        """Yields a HEIGHTxWIDTH list of floats representing the depth of each pixel in the current feed."""
        pass