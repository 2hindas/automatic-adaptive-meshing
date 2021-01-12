import SimPEG.electromagnetics.frequency_domain as fdem


def define_survey(frequencies, receiver_locations, source_locations, num_transmitters):
    """Defines a survey of a model with the receivers and the transmitters."""

    source_list = []
    for i in range(len(frequencies)):
        for j in range(num_transmitters):
            # Define receivers of different type at each location
            bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[j, :], "z", "real"
            )
            bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[j, :], "z", "imag"
            )
            receivers_list = [bzr_receiver, bzi_receiver]

            # Must define the transmitter properties and associated receivers
            source_list.append(
                fdem.sources.MagDipole(
                    receivers_list,
                    frequencies[i],
                    source_locations[j],
                    orientation="z",
                    moment=1,
                )
            )
    survey = fdem.Survey(source_list)
    return survey


def search_area_object(mesh, objct, factor=2):
    """Defines the search are in a mesh for a given object using a given factor that determines the
    search range."""

    cells = mesh.cell_centers
    cell_width_X = min(mesh.h[0])  # minimum cell width in x-direction
    cell_width_Y = min(mesh.h[1])  # minimum cell width in y-direction
    cell_width_Z = min(mesh.h[2])  # minimum cell width in z-direction
    Left_X = min(objct[:, 0])  # Left X
    Right_X = max(objct[:, 0])  # Right X
    Left_Y = min(objct[:, 1])  # Left Y
    Right_Y = max(objct[:, 1])  # Right Y
    Lower_Z = min(objct[:, 2])  # Lower Z
    Upper_Z = max(objct[:, 2])  # Upper Z
    search_area = cells[(cells[:, 0] > (Left_X - factor * cell_width_X)) & (
            cells[:, 0] < (Right_X + factor * cell_width_X))
                        & (cells[:, 1] > (Left_Y - factor * cell_width_Y)) & (
                                cells[:, 1] < (Right_Y + factor * cell_width_Y))
                        & (cells[:, 2] > (Lower_Z - factor * cell_width_Z)) & (
                                cells[:, 2] < (Upper_Z + factor * cell_width_Z))]

    return search_area


def search_area_receivers(mesh, receiver_locations, factor=3):
    """Defines the search area in a mesh for given receivers using a given factor that determines
    the search range."""

    cells = mesh.cell_centers
    cell_width_X = min(mesh.h[0])  # minimum cell width in x-direction
    cell_width_Y = min(mesh.h[1])  # minimum cell width in y-direction
    cell_width_Z = min(mesh.h[2])  # minimum cell width in z-direction
    Left_X = min(receiver_locations[:, 0])  # Left X
    Right_X = max(receiver_locations[:, 0])  # Right X
    Left_Y = min(receiver_locations[:, 1])  # Left Y
    Right_Y = max(receiver_locations[:, 1])  # Right Y
    Lower_Z = min(receiver_locations[:, 2])  # Lower Z
    Upper_Z = max(receiver_locations[:, 2])  # Upper Z
    search_area = cells[(cells[:, 0] > (Left_X - factor * cell_width_X)) & (
            cells[:, 0] < (Right_X + factor * cell_width_X))
                        & (cells[:, 1] > (Left_Y - factor * cell_width_Y)) & (
                                cells[:, 1] < (Right_Y + factor * cell_width_Y))
                        & (cells[:, 2] > (Lower_Z - factor * cell_width_Z)) & (
                                cells[:, 2] < (Upper_Z + factor * cell_width_Z))]

    return search_area

def search_area_sources(mesh, source_locations, factor=3):
    """Defines the search area in a mesh for given sources using a given factor that determines
    the search range."""

    cells = mesh.cell_centers
    cell_width_X = min(mesh.h[0])  # minimum cell width in x-direction
    cell_width_Y = min(mesh.h[1])  # minimum cell width in y-direction
    cell_width_Z = min(mesh.h[2])  # minimum cell width in z-direction
    Left_X = min(source_locations[:, 0])  # Left X
    Right_X = max(source_locations[:, 0])  # Right X
    Left_Y = min(source_locations[:, 1])  # Left Y
    Right_Y = max(source_locations[:, 1])  # Right Y
    Lower_Z = min(source_locations[:, 2])  # Lower Z
    Upper_Z = max(source_locations[:, 2])  # Upper Z
    search_area = cells[(cells[:, 0] > (Left_X - factor * cell_width_X)) & (
            cells[:, 0] < (Right_X + factor * cell_width_X))
                        & (cells[:, 1] > (Left_Y - factor * cell_width_Y)) & (
                                cells[:, 1] < (Right_Y + factor * cell_width_Y))
                        & (cells[:, 2] > (Lower_Z - factor * cell_width_Z)) & (
                                cells[:, 2] < (Upper_Z + factor * cell_width_Z))]

    return search_area


def get_ind_block(mesh, ind_active, coordinates):
    """Retreives the indices of a block object coordinates in a mesh."""

    return (
            (mesh.gridCC[ind_active, 0] <= coordinates[0][1])
            & (mesh.gridCC[ind_active, 0] >= coordinates[0][0])
            & (mesh.gridCC[ind_active, 1] <= coordinates[1][1])
            & (mesh.gridCC[ind_active, 1] >= coordinates[1][0])
            & (mesh.gridCC[ind_active, 2] <= coordinates[2][1])
            & (mesh.gridCC[ind_active, 2] >= coordinates[2][0])
    )


def get_ind_sphere(mesh, ind_active, origin, radius):
    """Retreives the indices of a sphere object coordintes in a mesh."""

    return (
            (mesh.gridCC[ind_active, 0] <= origin[0] + radius)
            & (mesh.gridCC[ind_active, 0] >= origin[0] - radius)
            & (mesh.gridCC[ind_active, 1] <= origin[1] + radius)
            & (mesh.gridCC[ind_active, 1] >= origin[1] - radius)
            & (mesh.gridCC[ind_active, 2] <= origin[2] + radius)
            & (mesh.gridCC[ind_active, 2] >= origin[2] - radius)
    )
