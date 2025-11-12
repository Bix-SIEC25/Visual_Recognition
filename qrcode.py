import cv2 as cv
import cvzone
from pyzbar import pyzbar as bar
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class QRCodeNode(Node):
    def __init__(self):
        super().__init__('qrcode_node')
        self.publisher_ = self.create_publisher(String, 'qrcode_data', 10)
        self.timer = self.create_timer(0.1, self.read_qrcode)  # 10x per second
        self.cap = cv.VideoCapture(1)
        self.get_logger().info('QRCode Node started!')

    def read_qrcode(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        result = bar.decode(frame)
        output = None

        for data in result:
            output = data.data.decode('utf-8')
            msg = String()
            msg.data = output
            self.publisher_.publish(msg)
            self.get_logger().info(f'QR Code detected: {output}')

        # Camera display
        cvzone.putTextRect(frame, 'QrCode Scanner', (190, 30), scale=2, thickness=2, border=2)
        if output:
            cvzone.putTextRect(frame, output, (40, 300), scale=2, thickness=2, border=2)
        cv.imshow('frame', frame)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = QRCodeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
