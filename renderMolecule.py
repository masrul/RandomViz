from elements import elements as elements
import bpy
import os
import sys
import math
import numpy as np


def drawCylinder(x1, y1, z1, x2, y2, z2, r=0.50):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=r, depth=dist, location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1)
    )

    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)

    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


def getMaterial(color, metallic, roughness, opacity):
    specularColor = color
    baseColor = color + (opacity,)
    diffuseColor = color + (opacity,)

    mat = bpy.data.materials.new(name="NewMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs["Metallic"].default_value = metallic * 2.0
    nodes["Principled BSDF"].inputs["Roughness"].default_value = roughness
    nodes["Principled BSDF"].inputs["Base Color"].default_value = baseColor
    mat.metallic = metallic
    mat.roughness = roughness
    mat.diffuse_color = diffuseColor
    mat.specular_color = specularColor

    return mat


class Molecule:
    def __init__(self, coordFile, colorType="vesta"):
        self.coordFile = coordFile
        self.colorType = colorType
        self.__initData()
        self.__readCoord()
        self.__setVdwRadius()
        self.__setColor()
        self.__genBonds()

    def __initData(self):
        self.natoms = 0
        self.nbonds = 0
        self.symbols = []
        self.x = []
        self.y = []
        self.z = []
        self.vdw_radii = []
        self.colors = []
        self.bonds = []

    def __readCoord(self):
        if self.coordFile.endswith(".pdb"):
            self.__readPDB()
        elif self.coordFile.endswith(".xyz"):
            self.__readXYZ()
        else:
            extension = os.path.splitext(self.coordFile)[-1]
            print("FatalError: Unsupported file format " + "(*" + extension + ")")
            print("Program quitting ....")
            sys.exit()

    def __setVdwRadius(self):
        print("setting radius ...")
        for symbol in self.symbols:
            symbol = symbol[0]
            radius = elements[symbol]["vdw_radius"]
            self.vdw_radii.append(radius)

    def __setColor(self):
        print("setting colors ...")
        if self.colorType == "vesta":
            colorType = "vesta_color"
        elif self.colorType == "cpk":
            colorType = "cpk_color"
        elif self.colorType == "jmol":
            colorType = "jmol_color"

        for symbol in self.symbols:
            symbol = symbol[0]
            color = elements[symbol][colorType]
            self.colors.append(color)

    def center(self):
        xcom = np.mean(self.x)
        ycom = np.mean(self.y)
        zcom = np.mean(self.z)

        self.x = [(xi - xcom) for xi in self.x]
        self.y = [(yi - ycom) for yi in self.y]
        self.z = [(zi - zcom) for zi in self.z]

    def customizeColor(self, selText, color):
        selType = selText.split()[0]

        if selType == "symbol":
            symbol = selText.split()[1]
            for iatom in range(self.natoms):
                if self.symbols[iatom] == symbol:
                    self.colors[iatom] = color

        elif selType == "index":
            index = [int(idx) for idx in selText.split()[1:]]
            for iatom in range(self.natoms):
                if iatom in index:
                    self.colors[iatom] = color

        else:
            print("selection-type(%s) is not implemented!" % selType)
            print("Program quitting ....")
            sys.exit()

    def __readXYZ(self):
        xyzFH = open(self.coordFile, "r")
        lines = xyzFH.readlines()
        xyzFH.close()

        self.natoms = int(lines[0])
        for line in lines[2:]:
            self.symbols.append(line.split()[0])
            self.x.append(float(line.split()[1]))
            self.y.append(float(line.split()[2]))
            self.z.append(float(line.split()[3]))

    def __readPDB(self):
        pdbFH = open(self.coordFile, "r")
        lines = pdbFH.readlines()
        pdbFH.close()

        for line in lines:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                self.symbols.append(line[12:16].strip())
                self.x.append(float(line[30:38]))
                self.y.append(float(line[38:46]))
                self.z.append(float(line[46:54]))

            if line.startswith("CONECT"):
                keys = line.split()

                iatom = int(keys[1]) - 1
                for i in range(2, len(keys)):
                    jatom = int(keys[i]) - 1

                    if iatom < jatom:
                        self.bonds.append((iatom, jatom))
                    else:
                        self.bonds.append((jatom, iatom))

        self.natoms = len(self.x)

        if len(self.bonds) != 0:
            self.bonds = list(set(self.bonds))
            self.nbonds = len(self.bonds)

        print("Bonds found: %d" % self.nbonds)

    def __genBonds(self):

        print("Generating bond list using VdW radius")
        if self.nbonds != 0:
            return

        maxBonds = 4

        for i in range(self.natoms - 1):
            ivdw = self.vdw_radii[i]
            BondCounts = 0
            for j in range(i + 1, self.natoms):
                jvdw = self.vdw_radii[j]
                rcut = 0.6 * (ivdw + jvdw)
                rcut = rcut * rcut
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                dz = self.z[i] - self.z[j]

                r = dx * dx + dy * dy + dz * dz
                if r <= rcut:
                    self.bonds.append((i, j))
                    self.nbonds += 1
                    BondCounts += 1
                if BondCounts == maxBonds:
                    break

    def getBondDirection(self, bondID, r_cyl):

        iatom = self.bonds[bondID][0]
        jatom = self.bonds[bondID][1]

        ir, ig, ib = self.colors[iatom]
        jr, jg, jb = self.colors[jatom]

        x1, y1, z1 = (self.x[iatom], self.y[iatom], self.z[iatom])
        x2, y2, z2 = (self.x[jatom], self.y[jatom], self.z[jatom])

        xmid = 0.50 * (x1 + x2)
        ymid = 0.50 * (y1 + y2)
        zmid = 0.50 * (z1 + z2)

        iloc = (
            0.50 * (x1 + xmid),
            0.50 * (y1 + ymid),
            0.50 * (z1 + zmid),
        )

        jloc = (
            0.50 * (x2 + xmid),
            0.50 * (y2 + ymid),
            0.50 * (z2 + zmid),
        )

        idx = xmid - x1
        idy = ymid - y1
        idz = zmid - z1

        jdx = x2 - xmid
        jdy = y2 - ymid
        jdz = z2 - zmid

        idist = math.sqrt(idx ** 2 + idy ** 2 + idz ** 2)
        jdist = math.sqrt(jdx ** 2 + jdy ** 2 + jdz ** 2)

        iphi = math.atan2(idy, idx)
        jphi = math.atan2(jdy, jdx)

        itheta = math.acos(idz / idist)
        jtheta = math.acos(jdz / jdist)

        iscale = (1.0, 1.0, idist * (1.0 / r_cyl) * 0.5)
        jscale = (1.0, 1.0, jdist * (1.0 / r_cyl) * 0.5)

        bondParams = {
            "color": [(ir, ig, ib), (jr, jg, jb)],
            "loc": (iloc, jloc),
            "dist": (idist, jdist),
            "phi": (iphi, jphi),
            "theta": (itheta, jtheta),
            "scale": (iscale, jscale),
        }

        return bondParams

    def drawAtoms(self, r_ref):
        bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=r_ref)
        sphere = bpy.context.object
        bpy.context.collection.objects.unlink(sphere)

        for iatom in range(self.natoms):
            scale = self.vdw_radii[iatom]

            copy = sphere.copy()
            copy.data = sphere.data.copy()

            copy.location.x = self.x[iatom]
            copy.location.y = self.y[iatom]
            copy.location.z = self.z[iatom]
            copy.scale = (scale, scale, scale)

            mat = getMaterial(
                color=self.colors[iatom], metallic=0.75, roughness=0.75, opacity=1.0
            )

            copy.active_material = mat
            bpy.context.collection.objects.link(copy)

    def drawBonds(self, r_cyl):
        bpy.ops.surface.primitive_nurbs_surface_cylinder_add(radius=r_cyl)
        cylinder = bpy.context.object
        bpy.context.collection.objects.unlink(cylinder)

        for ibond in range(self.nbonds):
            bondParams = self.getBondDirection(ibond, r_cyl)

            icopy = cylinder.copy()
            jcopy = cylinder.copy()

            ii = 0
            for copy in (icopy, jcopy):
                copy.data = cylinder.data.copy()
                copy.location = bondParams["loc"][ii]

                copy.rotation_euler[1] = bondParams["theta"][ii]
                copy.rotation_euler[2] = bondParams["phi"][ii]

                copy.scale = bondParams["scale"][ii]

                color = bondParams["color"][ii]
                mat = getMaterial(
                    color=color, metallic=0.75, roughness=0.75, opacity=1.0
                )
                copy.active_material = mat
                bpy.context.collection.objects.link(copy)
                ii += 1

    def draw(self):
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)

        r_ref = 0.2
        r_cyl = 0.90 * r_ref

        self.drawAtoms(r_ref)
        self.drawBonds(r_cyl)
        bpy.ops.object.shade_smooth()
        bpy.context.scene.objects.update()  # only once
        bpy.ops.object.select_all(action="DESELECT")


def main():
    mol = Molecule("syste.pdb")
    # mol = Molecule("FAU.pdb")
    mol.center()
    print("drawing ...")
    mol.draw()


main()
