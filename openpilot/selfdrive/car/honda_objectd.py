#!/usr/bin/env python3
"""Decode the 11th-gen Accord's tracked-object list for the UI only.

The car's driver-assist bus (panda bus 1, CAN FD) carries 16 object slots at 0x062..0x080, 64 bytes each, ~15 Hz.
Fields reverse-engineered from drive logs (see the commit message for how each was checked):
  byte 4      object id (stable for the life of a track)
  bytes 6-7   distance, little-endian, 0.00625 m/unit (checked against the camera lead: city ~27 m, highway ~100 m)
  bytes 8-9   lateral, little-endian, centre 0x6000, 0.01 m/unit, left positive (r = -1.00 against the camera lead)
  bytes 10-11 relative speed, little-endian, centre 0x600, ~0.0121 m/s per unit (r = 0.85 against range rate)
A zero distance means the slot is empty.

This only feeds the UI's top view, published on customReservedRawData1. It never reaches radard,
lead selection or longitudinal control, so driving behaviour is unchanged.
Payload: repeated little-endian records of (uint8 id, float32 dist_m, float32 lat_left_m, float32 vrel_mps).
"""
import struct

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog

OBJ_BUS = 1
OBJ_ADDRS = set(range(0x062, 0x082, 2))
DIST_SCALE = 0.00625
LAT_CENTRE, LAT_SCALE = 0x6000, 0.01
VREL_CENTRE, VREL_SCALE = 0x600, 0.0121
MAX_LAT = 15.0  # m; some slots carry other encodings with nonsense lateral values, drop those
MAX_DIST = 200.0
TRIGGER_ADDR = 0x080  # last slot of the cycle: publish once per full sweep
RECORD = struct.Struct('<Bfff')


def decode(dat: bytes):
  if len(dat) < 12:
    return None
  raw_dist = dat[6] | dat[7] << 8
  if raw_dist == 0:
    return None
  dist = raw_dist * DIST_SCALE
  lat = ((dat[8] | dat[9] << 8) - LAT_CENTRE) * LAT_SCALE
  vrel = ((dat[10] | dat[11] << 8) - VREL_CENTRE) * VREL_SCALE
  if not (0.5 < dist < MAX_DIST) or abs(lat) > MAX_LAT:
    return None
  return dat[4], dist, lat, vrel


def main():
  can_sock = messaging.sub_sock('can', conflate=False, timeout=100)
  pm = messaging.PubMaster(['customReservedRawData1'])
  slots: dict[int, tuple] = {}
  cloudlog.info("honda_objectd: started")

  while True:
    for msg in messaging.drain_sock(can_sock, wait_for_one=True):
      publish = False
      for c in msg.can:
        if c.src != OBJ_BUS or c.address not in OBJ_ADDRS:
          continue
        obj = decode(bytes(c.dat))
        if obj is None:
          slots.pop(c.address, None)
        else:
          slots[c.address] = obj
        publish |= c.address == TRIGGER_ADDR
      if publish:
        out = messaging.new_message(None, valid=True)
        out.customReservedRawData1 = b''.join(RECORD.pack(*o) for o in slots.values())
        pm.send('customReservedRawData1', out)


if __name__ == "__main__":
  main()
