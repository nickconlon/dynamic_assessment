import cv2
import numpy as np
import project.multiagent_configs as configs
import project.rendering_environment as env


def add_convex_hull(_image, _obstacles, _color):
    imgg = np.zeros_like(_image)
    _img = _image.copy()
    for obs in _obstacles:
        imgg = cv2.circle(imgg, (obs.xy[0] * 10, obs.xy[1] * 10), obs.r * 10, obs.color, thickness=2)

    gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    kernel_dilation = np.ones((4, 4), np.uint8)
    gray = cv2.dilate(gray, kernel_dilation, iterations=12)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    for i in range(len(contours)):
        _img = cv2.drawContours(_img, hull, i, _color, -1)

    alpha = 0.4
    _img = cv2.addWeighted(_img, alpha, _image, 1 - alpha, 0)

    return hull, contours, _img


if __name__ == '__main__':
    renderer = env.MultiAgentRendering([1])
    cp, zp = configs.read_scenarios(1)

    img = renderer.render(mode='rgb_array')
    img_copy = img.copy()
    _, _, img_copy = add_convex_hull(img_copy, zp, (22, 22, 138))
    _, _, img_copy = add_convex_hull(img_copy, cp, (255, 0, 0))

    out = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    cv2.imshow('t', out)
    cv2.waitKey(-1)

    renderer.change_event(new_craters=cp, new_zones=zp)
    cv2.imshow('t', cv2.cvtColor(renderer.render(mode='rgb_array'), cv2.COLOR_BGR2RGB))
    cv2.waitKey(-1)

