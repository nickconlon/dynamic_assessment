import zmq
import base64
import cv2
import numpy as np


class ZmqSubscriber:
    def __init__(self, ip, port, topic="", timeout=None, last_only=False, bind=False):
        self.ip = ip
        self.port = port
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        if timeout is not None:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        if last_only:
            self.socket.setsockopt(zmq.CONFLATE, 1)
        if bind:
            self.socket.bind("tcp://{}:{}".format(self.ip, self.port))
        else:
            self.socket.connect("tcp://{}:{}".format(self.ip, self.port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def receive(self):
        data = ""
        try:
            data = self.socket.recv_string()
        except Exception as e:
            print(e)
        return data

    def close(self):
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    #sub = ZmqSubscriber("localhost", "5557")
    data_sub = ZmqSubscriber("*", "5556", bind=True)
    #string = sub.receive()
    #jpg_original = base64.b64decode(string)
    #jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    #img = cv2.imdecode(jpg_as_np, flags=1)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)

    # Process 5 updates
    total_temp = 0
    for update_nbr in range(5):
        string = data_sub.receive()
        print(string)