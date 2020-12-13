import SimPEG.electromagnetics.frequency_domain as fdem


def define_survey(frequencies, receiver_locations, source_locations, num_transmitters):
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


