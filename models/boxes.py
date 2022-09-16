import numpy as np
import os
import math
import torch
from projectUtils import euler_to_matrix

class OBJ:
    def __init__(self, vertices, lines_index):

        self.vertices = vertices
        self.lines = [[line[0], line[1]] for line in lines_index]

    def save_obj(self, output_dir):
        with open(output_dir, 'w') as f:
            f.write('# OBJ file\n')
            for v in self.vertices:
                f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
            for l in self.lines:
                f.write('l {0} {1}\n'.format(int(l[0]), int(l[1])))

class Box3d:

    """
    Parameters:

        center:  list with shape (3,)
                 x, y, z coordinate for center of 3d-box

        dim: list with shape (3,)
             width, height, depth of 3d-box

        rot: float
             clockwise rotation angle around z-axis


    points orientation:

           pt3----------pt1
          / |           / |
         /  |          /  |
        pt5-|--------pt7  |
        |   |         |   |
        |   |         |   |
        |   pt2-------|--pt0
        |  /          |  /
        | /           | /
        pt4----------pt6

    eight-points:

        points[0]: right, rear, down  ;  points[1]: right, rear, top
        points[2]: left, rear, down   ;  points[3]: left, rear, top
        points[4]: left, front, down  ;  points[5]: left, front, top
        points[6]: right, front, down ;  points[7]: right, front, top

    six-faces:

        faces[0]:  bottom:  (pt0, pt2, pt4)
        faces[1]:  top:     (pt1, pt3, pt5)
        faces[2]:  rear:    (pt0, pt1, pt2)
        faces[3]:  left:    (pt2, pt3, pt4)
        faces[4]:  front:   (pt4, pt5, pt6)
        faces[5]:  right:   (pt6, pt7, pt0)

    orientation:
        X-right, y-front, z-down

    """

    # center: [x, y, z], dim: [w, h, l]
    def __init__(self, param):

        #center, dim, rot = param[:3], param[3:6], param[6]
        #param = param.cpu().detach().numpy()
        center, dim, rot = param[:, :, :3], param[:, :, 3:6], param[:, :, 6:7]

        self.center = center
        self.dim = dim
        self.rot = rot  # rotation angle around y-axis

        trans = self.center
        rot = self.rot

        #rot_mat = euler_to_matrix(torch.tensor([0, rot, 0], dtype=torch.float32, device='cuda:0'))

        # store eight-points
        self.points = []
        for i in [1, -1]:
            for j in [1, -1]:
                for k in [-1, 1]:
                    # point = np.copy(center)  # center: [x, y, z], dim: [w, h, l]
                    # point[0] = i * dim[0]
                    # point[1] = k * dim[1]
                    # point[2] = (j * i) * dim[2]
                    # self.points.append(point)

                    point = center.clone() # center: [x, y, z], dim: [w, h, l]
                    point = dim * torch.tensor([i, k, j*i], device=point.device)[None, None, ...]
                    # point[0] = i * dim[0]
                    # point[1] = k * dim[1]
                    # point[2] = (j * i) * dim[2]
                    self.points.append(point)
        #self.points = np.array(self.points)
        self.points = torch.cat(self.points, dim=0)

        from projectUtils import point_transform_local_to_world
        #self.points = point_transform_local_to_world(torch.tensor(self.points).cuda().unsqueeze(0), trans, rot)[0].cpu()
        self.points = point_transform_local_to_world(self.points, trans, rot).permute(0, 2, 1).squeeze().cpu().detach().numpy()

        # store twelve-lines
        self.lines = []
        for i in range(4):
            line = [2 * i, 2 * i + 1]
            self.lines.append(line)
        for i in range(8):
            line = [i, (i + 2) % 8]
            self.lines.append(line)
        self.lines = np.array(self.lines)

    def save_obj(self, out_dir):

        obj = OBJ(self.points[0], np.array(self.lines) + 1)
        obj.save_obj(out_dir)

    def convert_to_iou3d_format(self):
        points = np.copy(self.points)
        points[0] = self.points[3]
        points[1] = self.points[1]
        points[2] = self.points[0]
        points[3] = self.points[2]
        points[4] = self.points[5]
        points[5] = self.points[7]
        points[6] = self.points[6]
        points[7] = self.points[4]
        return points

class Boxes:
    def __init__(self, preds):
        """
        :param preds: [N, 7]
        """
        # N = preds.shape[0]
        # self.boxes = []
        # for i in range(N):
        #     self.boxes.append(Box3d(preds[i]))

        nParts = preds.shape[1]
        preds = torch.chunk(preds, nParts, dim=1)

        self.boxes = []
        for i in range(nParts):
            self.boxes.append(Box3d(preds[i]))

    def save_obj(self, path):
        points, lines = np.empty((0, 3)), np.empty((0, 2))
        for i, box in enumerate(self.boxes):
            points = np.append(points, box.points, axis=0)
            lines = np.append(lines, box.lines + i*8, axis=0)
            #lines += (box.lines + i*12)

        obj = OBJ(points, lines+1)
        obj.save_obj(path)


def nms_3d(preds, threshold=0.5):
    # based on largest volume
    boxes = Boxes(preds)
    boxes = boxes.boxes
    boxes.sort(key=lambda x: -torch.prod(x.dim))
    mask = [True] * len(boxes)

    for i, box in enumerate(boxes):
        if not mask[i]:
            continue
        for j, box2 in enumerate(boxes[i+1:]):
            if not mask[j]:
                continue
            from extern.pytorch3d.pytorch3d.ops import box3d_overlap
            _, iou_3d = box3d_overlap(torch.tensor(np.array([box.convert_to_iou3d_format()])),
                                      torch.tensor(np.array([box2.convert_to_iou3d_format()])))
            if iou_3d > threshold:
                mask[j] = False
    ret = preds[0][mask].unsqueeze(0)
    return ret


if __name__ == '__main__':
    # boxes = Boxes(torch.Tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 90/180 * np.pi], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 90/180 * np.pi]]))
    # boxes.save_obj('out.obj')
    preds = torch.tensor([[[ -1.6326,   0.2987,   0.8947,   3.4665,   7.5890,   5.2823,  -6.5871],
                             [ -3.5177,  -2.8600,   5.7947,   3.7750,   7.5632,   6.9629, -15.1123],
                             [  2.7870,   7.9580,  -1.8690,   4.1614,  11.4128,   4.0250,   4.3629],
                             [ -0.2533,  -5.0189,   1.3366,   1.9037,   7.5709,   2.3022,  -0.8998],
                             [  2.4009,   0.7081,  -2.4128,   3.5692,   7.6509,   5.6303,  -3.7983],
                             [  1.9965,  -1.2916,  -0.8033,   4.0430,   4.2715,   4.2745,  -0.3570],
                             [  0.4659,  -0.8109,   0.4332,   7.1752,   4.6992,   4.3783,   1.3861],
                             [ -5.2215,   5.4337,  -1.2980,   7.0939,   5.6666,   2.6346,   6.5449],
                             [  3.1034,  -0.5812,  -1.5850,  16.9340,   4.7562,   3.8494,   8.3605],
                             [ -1.5972,   0.3287,  -1.2872,   2.8456,   9.4504,   4.5568,   5.0369]]], device='cuda:0')

    preds = nms_3d(preds, 0.1)
    boxes = Boxes(preds)

    boxes.save_obj('test_out2.obj')
