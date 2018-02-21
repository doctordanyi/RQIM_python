"""Run RQIM experiments. Placeholder for now"""
import json
import lib.structures.quad as quad

quads = [quad.create_random(0.5, 0.2, 1, 0, 2) for i in range(100)]
file = open("quads.json", "w")
json.dump(quads, file, cls=quad.QuadEncoder, separators=(',', ':'), indent=4)

