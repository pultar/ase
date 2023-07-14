

from ase import Atoms
from ase.io.utils import PlottingVariables
import numpy as np

atoms = Atoms('H3',
    positions=[
        [0.0,0,0],
        [0.3,0,0],
        [0.8,0,0]],
    cell = [1,1,1],
    pbc=True)


rotation = '0x, 45y, 0z'


generic_projection_settings = {
    'rotation': rotation,
    #'radii': len(atoms) * [0.2],
    'show_unit_cell': 2}

pl = PlottingVariables(atoms, **generic_projection_settings)










def test_set_bbox():
    rotation = '0x, 0y, 0z'

    generic_projection_settings = {
        'rotation': rotation,
        'bbox':(0,0,1,1)}

    pl = PlottingVariables(atoms, **generic_projection_settings)

    scale = pl.scale

    camera_location = pl.get_image_plane_center()


def test_camera_directions():

    rotation = '0x, 45y, 0z'

    generic_projection_settings = {   'rotation': rotation}

    pl = PlottingVariables(atoms, **generic_projection_settings)


    camdir = pl.get_camera_direction()
    up     = pl.get_camera_up()
    right  = pl.get_camera_right()

    assert np.allclose(camdir.T @ up, 0)
    assert np.allclose(camdir.T @ right, 0)
    assert np.allclose(right.T @ up, 0)


    r22 = np.sqrt(2)/2
    assert np.allclose(camdir, [ r22, 0, -r22])
    assert np.allclose(up, [0, 1, 0])
    assert np.allclose(right, [ r22, 0, r22])
