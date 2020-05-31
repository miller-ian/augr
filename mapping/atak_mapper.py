import os
from mapping.publisher import CoT
import logging

import time

# TODO clean this up
ATAK_IP = os.getenv('ATAK_IP', '192.168.1.39')
ATAK_PORT = int(os.getenv('ATAK_PORT', '6969'))
ATAK_PROTO = os.getenv('ATAK_PROTO', 'UDP')

# params = {  # SWX parking lot
#     "lat": 27.957261,
#     "lon": -82.436587,
#     "uid": "Ian Miller",
#     "identity": "hostile",
#     "dimension": "land-unit",
#     "entity": "military",
#     "type": "U-C"
# #    "type": "U-C-R-H"
# }

def publish_detection(lat, lon, name='person', identity='hostile', dimension='land-unit', entity='military', mtype='U-C'):
    params = {
        "lat": 28.752345,
        "lon": -81.390342,
        "uid": name,
        "identity": identity,
        "dimension": dimension,
        "entity": entity,
        "type": mtype
    }

    cot = CoT.CursorOnTarget()
    cot_xml = cot.atoms(params).encode()

    logging.debug('CoT XML: {}'.format(cot_xml))
    print("pushing to", str(lat), ",", str(lon))
    logging.info('Pushing detection {}/{}/{} at {} , {} to ATAK'.format(name, identity, dimension, lat, lon))

    if ATAK_PROTO == "TCP":
      sent = cot.pushTCP(ATAK_IP, ATAK_PORT, cot_xml)
    else:
      sent = cot.pushUDP(ATAK_IP, ATAK_PORT, cot_xml)

    logging.info(str(sent) + " bytes sent to " + ATAK_IP + " on port " + str(ATAK_PORT))