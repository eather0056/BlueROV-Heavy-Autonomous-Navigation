import rclpy
from rclpy.node import Node
import torch
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix, Imu, Image, LaserScan
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from torch.autograd import Variable
import torch.nn as nn
import cv2
import numpy as np
import argparse
import time
import torchvision.transforms as transforms
import tf2_ros as tf
from options.train_options import TrainOptions
from models.models import create_model

IMAGE_HIST = 4
dtype = torch.cuda.FloatTensor

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
model = create_model(opt)
model.switch_to_eval()

class UWsimWorld(Node):
    def __init__(self):
        super().__init__('uwsim')
        self.goal_table = np.array([[-15, 17]])  # goal for shipwreck obstacle
        self.cmd_vel = self.create_publisher(TwistStamped, 'g500/velocityCommand', 10)
        self.gps = self.create_subscription(NavSatFix, 'g500/gps', self.GetGPS, 10)
        self.orientation = self.create_subscription(Imu, 'g500/imu', self.GetOri, 10)
        self.rgb_image_sub = self.create_subscription(Image, 'g500/camera1', self.RGBImageCallBack, 10)
        self.echo_sounder_sub = self.create_subscription(LaserScan, '/g500/multibeam', self.MultibeamCallBack, 10)
        self.action_table = [-np.pi / 12, -np.pi / 24, 0., np.pi / 24, np.pi / 12]
        self.depth_image_size = [160, 128]
        self.rgb_image_size = [512, 384]
        self.bridge = CvBridge()
        self.i = 0
        self.cur_pos = None
        self.R2G = None
        self.euler = None
        self.rob_ori = None
        self.rgb_image = None
        self.multibeam = None

    def GetGPS(self, gps):
        self.cur_pos = np.array([-gps.latitude, gps.longitude])
        self.R2G = self.goal - self.cur_pos

    def GetPosition(self):
        return self.cur_pos, self.goal

    def choose_goal(self):
        self.goal = self.goal_table[self.i]
        self.i += 1

    def GetOri(self, orientation):
        quaternion = (orientation.orientation.x, orientation.orientation.y, orientation.orientation.z,
                      orientation.orientation.w)
        self.euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = -self.euler[2]
        if yaw < 0:
            yaw = yaw + 2 * np.pi
        self.rob_ori = np.array([np.cos(yaw), np.sin(yaw)])

    def Goal(self):
        distance = np.sqrt(self.R2G[0] ** 2 + self.R2G[1] ** 2) / 4
        angle = np.arccos(self.R2G.dot(self.rob_ori) / np.sqrt((self.R2G.dot(self.R2G)) * np.sqrt(self.rob_ori.dot(self.rob_ori)))) * 1.5

        if self.rob_ori[0] > 0 and (self.rob_ori[1] / self.rob_ori[0]) * self.R2G[0] > self.R2G[1]:
            angle = -angle
        elif self.rob_ori[0] < 0 and (self.rob_ori[1] / self.rob_ori[0]) * self.R2G[0] < self.R2G[1]:
            angle = -angle
        elif self.rob_ori[0] == 0:
            if self.rob_ori[1] > 0 and self.R2G[0] > 0:
                angle = -angle
            elif self.rob_ori[1] < 0 and self.R2G[0] < 0:
                angle = -angle

        goal = np.array([distance, angle])
        self.get_logger().info(f"goal: {goal}")
        return goal

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def MultibeamCallBack(self, multibeam):
        self.multibeam = multibeam

    def Multibeam(self):
        return self.multibeam.ranges

    def GetRGBImageObservation(self):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        cv2.imwrite("img.png", cv_img)
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_img = np.float32(cv_img) / 255
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        cv_resized_img = cv2.cvtColor(cv_resized_img, cv2.COLOR_BGR2RGB)
        return cv_resized_img

    def Control(self, forward, twist):
        move_cmd = TwistStamped()
        move_cmd.twist.linear.x = forward
        move_cmd.twist.linear.y = 0.
        move_cmd.twist.linear.z = 0.
        move_cmd.twist.angular.x = 0.
        move_cmd.twist.angular.y = 0.
        move_cmd.twist.angular.z = self.action_table[twist]
        self.cmd_vel.publish(move_cmd)


class DDDQN(nn.Module):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (10, 14), (8, 8), padding=(1, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1))

        self.img_goal_adv1 = nn.Linear(8 * 10 * 64, 512)
        self.img_goal_val1 = nn.Linear(8 * 10 * 64, 512)
        self.fc_goal = nn.Linear(4 * 2, 64)

        self.img_goal_adv2 = nn.Linear(576, 512)
        self.img_goal_val2 = nn.Linear(576, 512)
        self.img_goal_adv3 = nn.Linear(512, 5)
        self.img_goal_val3 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x, goal):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = self.relu(self.img_goal_adv1(x))
        val = self.relu(self.img_goal_val1(x))

        goal = goal.view(goal.size(0), -1)
        goal = self.relu(self.fc_goal(goal))

        adv = torch.cat((adv, goal), 1)
        val = torch.cat((val, goal), 1)
        adv = self.relu(self.img_goal_adv2(adv))
        val = self.relu(self.img_goal_val2(val))

        adv = self.img_goal_adv3(adv)
        val = self.img_goal_val3(val).expand(x.size(0), 5)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 5)
        return x


def DRL():
    rclpy.init()
    time_start = time.time()
    bridge = CvBridge()
    depth_img_pub = rclpy.create_publisher(Image, 'depth', 10)

    online_net = DDDQN()
    resume_file_online = '../DDDQN_goal_ppo/online_with_noise_64_small_shaped.pth.tar'
    checkpoint_online = torch.load(resume_file_online)
    online_net.load_state_dict(checkpoint_online['state_dict'])
    episode_number = checkpoint_online['episode']
    online_net.cuda()

    env = UWsimWorld()
    env.choose_goal()
    env.get_logger().info('Environment initialized')

    rclpy.sleep(2)
    rate = rclpy.Rate(3)

    with torch.no_grad():
        while rclpy.ok():
            t = 0
            rgb_img_t1 = env.GetRGBImageObservation()
            goal_t1 = env.Goal()

            input_img = torch.from_numpy(np.transpose(rgb_img_t1, (2, 0, 1))).contiguous().float()
            input_img = input_img.unsqueeze(0)
            input_img = Variable(input_img.cuda())
            pred_log_depth = model.netG.forward(input_img)

            depth_img_t1 = torch.exp(pred_log_depth)
            depth_img_cpu = depth_img_t1[0].data.squeeze().cpu().numpy().astype(np.float32)
            depth_img_cpu = bridge.cv2_to_imgmsg(depth_img_cpu, "passthrough")
            depth_img_pub.publish(depth_img_cpu)

            depth_img_t1 *= 5
            depth_img_t1[depth_img_t1 <= 2] -= .5
            depth_img_t1[depth_img_t1 >= 5.] = 0.0
            depth_img_t1 = nn.functional.interpolate(depth_img_t1, size=(128, 160))
            depth_img_t1 = torch.squeeze(depth_img_t1, 1)
            depth_imgs_t1 = torch.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), dim=1)
            goals_t1 = np.stack((goal_t1, goal_t1, goal_t1, goal_t1), axis=0)

            while rclpy.ok():
                rclpy.spin_once(env)
                multibeam = env.Multibeam()
                rgb_img_t1 = env.GetRGBImageObservation()
                goal_t1 = env.Goal()
                if goal_t1[0] < .25:
                    env.choose_goal()

                rgb_img_t1 = cv2.resize(rgb_img_t1, (512, 384), interpolation=cv2.INTER_LINEAR)
                rgb_img_t1 = cv2.cvtColor(rgb_img_t1, cv2.COLOR_BGR2RGB)

                input_img = torch.from_numpy(np.transpose(rgb_img_t1, (2, 0, 1))).contiguous().float()
                input_img = input_img.unsqueeze(0)
                input_img = Variable(input_img.cuda())

                goal = goal_t1
                pred_log_depth = model.netG.forward(input_img)
                depth_img_t1 = torch.exp(pred_log_depth)
                depth_img_t1 = nn.functional.interpolate(depth_img_t1, size=(128, 160))

                depth_img_cpu = depth_img_t1.permute(0, 2, 3, 1)
                depth_img_cpu = depth_img_cpu[0].data.squeeze().cpu().numpy().astype(np.float32)
                depth_img_cpu = bridge.cv2_to_imgmsg(depth_img_cpu, "passthrough")
                depth_img_pub.publish(depth_img_cpu)

                depth_imgs_t1 = torch.cat((depth_img_t1, depth_imgs_t1[:, :(IMAGE_HIST - 1), :, :]), 1)
                depth_imgs_t1_cuda = Variable(depth_imgs_t1.type(dtype))
                goal_t1 = np.reshape(goal_t1, (1, 2))
                goals_t1 = np.append(goal_t1, goals_t1[:(IMAGE_HIST - 1), :], axis=0)
                goals_t1_cuda = goals_t1[np.newaxis, :]
                goals_t1_cuda = torch.from_numpy(goals_t1_cuda)
                goals_t1_cuda = Variable(goals_t1_cuda.type(dtype))

                Q_value_list = online_net(depth_imgs_t1_cuda, goals_t1_cuda)
                Q_value_list = Q_value_list[0]
                Q_value, action = torch.max(Q_value_list, 0)

                if multibeam[50] > 1.2 and multibeam[40] > 1.2 and multibeam[60] > 1.2:
                    env.Control(0.25, action)
                else:
                    for _ in range(40000):
                        env.Control(0, 4)
                    distance_right = env.Multibeam()[50]
                    for _ in range(80000):
                        env.Control(0, 0)
                    distance_left = env.Multibeam()[50]

                    if distance_right > distance_left:
                        for _ in range(100000):
                            env.Control(0, 4)
                    else:
                        for _ in range(20000):
                            env.Control(0, 0)

                if t % 5 == 0:
                    with open("path_shipwreck_megaDRL.txt", "a") as path_file:
                        cur_pos, goal_pos = env.GetPosition()
                        data = [str(cur_pos[0]), ' ', str(cur_pos[1]), "\n"]
                        path_file.writelines(data)

                t += 1
                rate.sleep()
                if goal[0] < 1:
                    time_end = time.time()
                    print(f"spend time: {time_end - time_start} s")


def main():
    DRL()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
