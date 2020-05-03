# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:07:33 2020

@author: liamc
"""

import os
import time
import CoT

ATAK_IP = os.getenv('ATAK_IP', '239.2.3.1')
ATAK_PORT = int(os.getenv('ATAK_PORT', '6969'))
ATAK_PROTO = os.getenv('ATAK_PROTO', 'UDP')


"""
latitude must be a number

longitude must be a number

uid must be a string, represents the name of the CoT point

identity may be one of: pending, unknown, friend, neutral, hostile, assumed-friend, suspect

dimension may be one of: unknown, space, air, land-unit, land-equipment, sea-surface, land-installation, subsurface (or sea-subsurface), other

entity may be: military, civilian

type is MIL-STD-2525 function code in CoT format (single letters separated by hyphens), will be appended as-is to CoT Type string


Sample input data:
    
    params = {
	"lat": 30.0090027,
	"lon": -85.9578735,
   "uid": "Ian Miller",
	"identity": "hostile",
	"dimension": "land-unit"
	"entity": "military",
	"type": "E-V-A-T"
    }

"""
def pushCoT(latitude, longitude, uid, identity, dimension, entity, unitType):
    
    params = {  # SWX parking lot
    "lat": latitude,
    "lon": longitude,
    "uid": uid,
    "identity": identity,
    "dimension": dimension,
    "entity": entity,
    "type": unitType
    }
    
    print("Params:\n" + str(params))
    cot = CoT.CursorOnTarget()
    cot_xml = cot.atoms(params).encode()


    print("\nXML message:")
    print(cot_xml)
    
    print("\nPushing to ATAK...")
    #UDP is the default ATAK_PROTO
    if ATAK_PROTO == "TCP":
      sent = cot.pushTCP(ATAK_IP, ATAK_PORT, cot_xml)
    else:
      sent = cot.pushUDP(ATAK_IP, ATAK_PORT, cot_xml)
    print(str(sent) + " bytes sent to " + ATAK_IP + " on port " + str(ATAK_PORT))