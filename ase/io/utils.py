import numpy as np
from math import sqrt
from itertools import islice

from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors

def get_cell_vertex_points(cell, disp=(0.0,0.0,0.0)):
    ''' returns 8x3 list of the cell vertex coordinates'''
    cell_vertices = np.empty((2, 2, 2, 3))
    displacement = np.array(disp)
    for c1 in range(2):
        for c2 in range(2):
            for c3 in range(2):
                cell_vertices[c1, c2, c3] = np.dot([c1, c2, c3],
                                                   cell) + displacement
    cell_vertices.shape = (8, 3)
    return cell_vertices

def update_line_order_for_atoms(L, T, D, atoms, radii):
    # why/how does this happen before the camera rotation???
    R = atoms.get_positions()
    r2 = radii**2
    for n in range(len(L)):
        d = D[T[n]]
        if ((((R - L[n] - d)**2).sum(1) < r2) &
                (((R - L[n] + d)**2).sum(1) < r2)).any():
            T[n] = -1
    return T


class PlottingVariables:
    # removed writer - self
    def __init__(self, atoms, rotation='', show_unit_cell=2,
                 radii=None, bbox=None, colors=None, scale=20,
                 maxwidth=500, extra_offset=(0., 0.),
                 auto_bbox_size = 1.05):

        assert show_unit_cell in (0,1,2,3)
        '''
        show_unit_cell: 0 cell is not shown, 1 cell is shown, 2 cell is shown
                        and bounding box is computed to fit atoms and cell, 3
                        bounding box is fixed to cell only.
        '''

        self.show_unit_cell = show_unit_cell
        self.numbers = atoms.get_atomic_numbers()
        self.maxwidth = maxwidth
        self.atoms = atoms
        self.extra_offset = extra_offset
        self.auto_bbox_size=auto_bbox_size

        self.colors = colors
        if colors is None:
            ncolors = len(jmol_colors)
            self.colors = jmol_colors[self.numbers.clip(max=ncolors - 1)]

        if radii is None:
            radii = covalent_radii[self.numbers]
        elif isinstance(radii, float):
            radii = covalent_radii[self.numbers] * radii
        else:
            radii = np.array(radii)
        self.d = 2 * scale * radii
        self.radii = radii

        natoms = len(atoms)
        # will be overwritten if bbox is set
        self.scale = scale
        self.offset = np.zeros(3)

        if isinstance(rotation, str):
            rotation = rotate(rotation)
        self.rotation = rotation
        self.updated_image_plane_offset_and_size( bbox=bbox)
        # the old arcane stuff has been pushed into this function.
        self.update_patch_and_line_vars()

        # no displacement since it's a vector
        cell_vec_im = np.dot(atoms.get_cell(), rotation)
        cell_vec_im *= scale
        self.cell = cell_vec_im
        self.natoms = natoms
        self.constraints = atoms.constraints
        # extension for partial occupancies
        self.frac_occ = False
        self.tags = None
        self.occs = None

        try:
            self.occs = atoms.info['occupancy']
            self.tags = atoms.get_tags()
            self.frac_occ = True
        except KeyError:
            pass


    def update_patch_and_line_vars(self):
        '''Updates all the line and path stuff that is still in obvious, this
        function should be deprecated if nobody can understand why it's features
        exist'''
        cell = self.atoms.get_cell()
        disp = self.atoms.get_celldisp().flatten()
        positions = self.atoms.get_positions()

        if self.show_unit_cell > 0:
            L, T, D = cell_to_lines(self, cell)
            cell_verts_in_atom_coords = get_cell_vertex_points(cell, disp)
            cell_vertices = self.to_image_plane_positions(cell_verts_in_atom_coords)
            T = update_line_order_for_atoms(L, T, D, self.atoms,  self.radii)
            # D are a positions in the image plane, not sure why it's setup like this
            D = (self.to_image_plane_positions(D)+self.offset)[:, :2]
            positions = np.concatenate((positions, L) , axis=0)
        else:
            L = np.empty((0, 3))
            T = None
            D = None
            cell_vertices = None
        # just a rotations and scaling since offset is currently [0,0,0]
        positions = self.to_image_plane_positions(positions)
        self.positions = positions
        self.D = D # list of 2D cell points in the imageplane without the offset
        self.T = T # integers, probably z-order for lines?
        self.cell_vertices = cell_vertices


    def updated_image_plane_offset_and_size(self, bbox=None):
        if bbox is None:
            im_high, im_low =  self.get_bbox_from_atoms(self.atoms, self.d/2)
            if self.show_unit_cell in (2,3):
                cell = self.atoms.get_cell()
                disp = self.atoms.get_celldisp().flatten()
                cv_high, cv_low = self.get_bbox_from_cell(cell, disp)
                if self.show_unit_cell == 2:
                    im_low  = np.minimum(im_low,  cv_low)
                    im_high = np.maximum(im_high, cv_high)
                else:
                    im_low  = cv_low
                    im_high = cv_high
            middle = (im_high + im_low) / 2
            im_size = self.auto_bbox_size * (im_high - im_low)
            w = im_size[0]
            h = im_size[1]
            aspect = h/w
            if w > self.maxwidth:
                w = self.maxwidth
                h = aspect * w
                self.scale *= w / im_size[0]
            offset = np.array([ middle[0] - w / 2, middle[1] - h / 2, 0])
        else:
            w = (bbox[2] - bbox[0]) * scale
            h = (bbox[3] - bbox[1]) * scale
            offset = np.array([bbox[0], bbox[1], 0]) * scale

        self.offset = offset
        # why does the picture size change with extra_offset? seems like a bug
        self.w = w + self.extra_offset[0]
        self.h = h + self.extra_offset[1]
        # allows extra_offset to be 2D or 3D
        self.offset[:len(self.extra_offset)] -= np.array(self.extra_offset)

    def to_image_plane_positions(self, positions):
        im_positions = np.dot(positions, self.rotation)*self.scale - self.offset
        return im_positions

    def to_atom_positions(self, im_positions):
        positions = np.dot( (im_positions + self.offset)/self.scale, self.rotation.T)
        return positions

    def get_bbox_from_atoms(self, atoms, im_radii):
        im_positions = self.to_image_plane_positions( atoms.get_positions())
        im_low  = (im_positions - im_radii[:, None]).min(0)
        im_high = (im_positions + im_radii[:, None]).max(0)
        return im_high, im_low

    def get_bbox_from_cell(self, cell, disp=(0.0,0.0,0.0)):
        displacement = np.array(disp)
        cell_verts_in_atom_coords = get_cell_vertex_points(cell, displacement)
        cell_vertices = self.to_image_plane_positions(cell_verts_in_atom_coords)
        im_low = cell_vertices.min(0)
        im_high = cell_vertices.max(0)
        return im_high, im_low

    def get_image_plane_center(self):
        return self.to_atom_positions(np.array([0,0,0]))

    def get_camera_direction(self):
        c0 = self.get_image_plane_center()
        c1 = self.to_atom_positions(np.array([0,0,-1]))
        camera_direction = c1-c0
        return camera_direction/np.linalg.norm(camera_direction)

    def get_camera_up(self):
        c0 = self.get_image_plane_center()
        c1 = self.to_atom_positions(np.array([0,1,0]))
        camera_direction = c1-c0
        return camera_direction/np.linalg.norm(camera_direction)

    def get_camera_right(self):
        c0 = self.get_image_plane_center()
        c1 = self.to_atom_positions(np.array([1,0,0]))
        camera_direction = c1-c0
        return camera_direction/np.linalg.norm(camera_direction)


def cell_to_lines(writer, cell):
    # XXX this needs to be updated for cell vectors that are zero.
    # Cannot read the code though!  (What are T and D? nn?)
    nlines = 0
    nsegments = []
    for c in range(3):
        d = sqrt((cell[c]**2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n

    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    return positions, T, D


def make_patch_list(writer):
    from matplotlib.path import Path
    from matplotlib.patches import Circle, PathPatch, Wedge

    indices = writer.positions[:, 2].argsort()
    patch_list = []
    for a in indices:
        xy = writer.positions[a, :2]
        if a < writer.natoms:
            r = writer.d[a] / 2
            if writer.frac_occ:
                site_occ = writer.occs[str(writer.tags[a])]
                # first an empty circle if a site is not fully occupied
                if (np.sum([v for v in site_occ.values()])) < 1.0:
                    # fill with white
                    fill = '#ffffff'
                    patch = Circle(xy, r, facecolor=fill,
                                   edgecolor='black')
                    patch_list.append(patch)

                start = 0
                # start with the dominant species
                for sym, occ in sorted(site_occ.items(),
                                       key=lambda x: x[1],
                                       reverse=True):
                    if np.round(occ, decimals=4) == 1.0:
                        patch = Circle(xy, r, facecolor=writer.colors[a],
                                       edgecolor='black')
                        patch_list.append(patch)
                    else:
                        # jmol colors for the moment
                        extent = 360. * occ
                        patch = Wedge(
                            xy, r, start, start + extent,
                            facecolor=jmol_colors[atomic_numbers[sym]],
                            edgecolor='black')
                        patch_list.append(patch)
                        start += extent

            else:
                if ((xy[1] + r > 0) and (xy[1] - r < writer.h) and
                        (xy[0] + r > 0) and (xy[0] - r < writer.w)):
                    patch = Circle(xy, r, facecolor=writer.colors[a],
                                   edgecolor='black')
                    patch_list.append(patch)
        else:
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                patch = PathPatch(Path((xy + hxy, xy - hxy)))
                patch_list.append(patch)
    return patch_list


class ImageChunk:
    """Base Class for a file chunk which contains enough information to
    reconstruct an atoms object."""

    def build(self, **kwargs):
        """Construct the atoms object from the stored information,
        and return it"""
        pass


class ImageIterator:
    """Iterate over chunks, to return the corresponding Atoms objects.
    Will only build the atoms objects which corresponds to the requested
    indices when called.
    Assumes ``ichunks`` is in iterator, which returns ``ImageChunk``
    type objects. See extxyz.py:iread_xyz as an example.
    """

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, index=None, **kwargs):
        if isinstance(index, str):
            index = string2index(index)

        if index is None or index == ':':
            index = slice(None, None, None)

        if not isinstance(index, (slice, str)):
            index = slice(index, (index + 1) or None)

        for chunk in self._getslice(fd, index):
            yield chunk.build(**kwargs)

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd),
                              indices.start, indices.stop,
                              indices.step)
        except ValueError:
            # Negative indices.  Go through the whole thing to get the length,
            # which allows us to evaluate the slice, and then read it again
            if not hasattr(fd, 'seekable') or not fd.seekable():
                raise ValueError(('Negative indices only supported for '
                                  'seekable streams'))

            startpos = fd.tell()
            nchunks = 0
            for chunk in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator


def verify_cell_for_export(cell, check_orthorhombric=True):
    """Function to verify if the cell size is defined and if the cell is

    Parameters:

    cell: cell object
        cell to be checked.

    check_orthorhombric: bool
        If True, check if the cell is orthorhombric, raise an ``ValueError`` if
        the cell is orthorhombric. If False, doesn't check if the cell is
        orthorhombric.

    Raise a ``ValueError`` if the cell if not suitable for export to mustem xtl
    file or prismatic/computem xyz format:
        - if cell is not orthorhombic (only when check_orthorhombric=True)
        - if cell size is not defined
    """

    if check_orthorhombric and not cell.orthorhombic:
        raise ValueError('To export to this format, the cell needs to be '
                         'orthorhombic.')
    if cell.rank < 3:
        raise ValueError('To export to this format, the cell size needs '
                         'to be set: current cell is {}.'.format(cell))


def verify_dictionary(atoms, dictionary, dictionary_name):
    """
    Verify a dictionary have a key for each symbol present in the atoms object.

    Parameters:

    dictionary: dict
        Dictionary to be checked.


    dictionary_name: dict
        Name of the dictionary to be displayed in the error message.

    cell: cell object
        cell to be checked.


    Raise a ``ValueError`` if the key doesn't match the atoms present in the
    cell.
    """
    # Check if we have enough key
    for key in set(atoms.symbols):
        if key not in dictionary:
            raise ValueError('Missing the {} key in the `{}` dictionary.'
                             ''.format(key, dictionary_name))
