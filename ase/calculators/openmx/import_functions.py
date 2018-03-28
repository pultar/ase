"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import os
import subprocess
from numpy import cos, sin, arctan, sqrt, pi, isfinite


def remove_infinities(sequence):
    for i in range(len(sequence)):
        if not isfinite(sequence[i]):
            sequence[i] = 0
    return sequence


def read_nth_to_last_value(line='\n', n=1):
    if line == '':
        return 0
    i = 0
    for j in range(n):
        i -= 1
        while line.split(' ')[i] == '' or line.split(' ')[i] == '\n':
            i -= 1

    return line.split(' ')[i].split('\n')[0]


def input_command(calc, executable_name, input_files, argument_format='%s'):
    if type(input_files) is list:
        input_files = tuple(input_files)
    command = executable_name + ' ' + argument_format % input_files
    olddir = os.getcwd()
    try:
        os.chdir(calc.directory)
        error_code = subprocess.call(command, shell=True)
    finally:
        os.chdir(olddir)
    if error_code:
        raise RuntimeError('%s returned an error: %d' %
                           (executable_name, error_code))


def spherical_polar_to_cartesian(r, theta, phi):  # degrees
    rho = r * sin(theta * pi / 180)
    x = rho * cos(phi * pi / 180)
    y = rho * sin(phi * pi / 180)
    z = r * cos(theta * pi / 180)
    return x, y, z


def cartesian_to_spherical_polar(x, y, z):
    rho2 = x * x + y * y
    r = sqrt(rho2 + z * z)
    rho = sqrt(rho2)
    theta = arctan(z / rho)
    if z < 0:
        theta += pi
    phi = arctan(y / float(x))
    if x < 0:
        if y < 0:
            phi -= pi
        if y > 0:
            phi += pi
    return r, theta * 180 / pi, phi * 180 / pi  # degrees
