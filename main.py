from structures.quad import Quad
from graphics.renderer import Renderer
import numpy as np
import math as m
import cv2

quad = Quad.create_from_params(0.5, m.pi / 4, m.pi / 4, m.sqrt(2) / 4, m.sqrt(2) / 4, 0)
rend = Renderer()

for a in np.arange(0.05, 0.7, 0.06):
    generated = 0
    while generated < 3:
        q = Quad.create_random(a, 0.2, 1, 0.3, 2)
        if q.is_valid():
            generated = generated + 1
            img = rend.render(q)
            cv2.imshow("Quad", img)
            cv2.waitKey(0)



