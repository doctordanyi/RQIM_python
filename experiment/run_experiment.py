"""Run RQIM experiments."""
import json, cv2
import lib.structures.quad as quad
import lib.graphics.renderer as renderer

quads = [quad.create_random(0.5, 0.2, 1, 0, 2) for i in range(100)]
file = open("quads.json", "w")
json.dump(quads, file, cls=quad.QuadEncoder, separators=(',', ':'), indent=4)

rend = renderer.Renderer()
i = 0
for q in quads:
    img = rend.render(q)
    cv2.imwrite("out/quad" + str(i) + ".png", img)
    i += 1
