"""
Transfer attacks package
"""

from .attacks import FGSM, PGD, CW, get_attack, create_transfer_model

__all__ = ['FGSM', 'PGD', 'CW', 'get_attack', 'create_transfer_model']

"""
Transfer attack implementations
""" 