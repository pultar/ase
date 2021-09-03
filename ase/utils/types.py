from typing import Sequence, TypeVar

__all__ = ['Sequence2D', 'Sequence3D', 'Sequence4D', 'Sequence5D']


T = TypeVar('T')


Sequence2D = Sequence[Sequence[T]]
Sequence3D = Sequence[Sequence[Sequence[T]]]
Sequence4D = Sequence[Sequence[Sequence[Sequence[T]]]]
Sequence5D = Sequence[Sequence[Sequence[Sequence[Sequence[T]]]]]
