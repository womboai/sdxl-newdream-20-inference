from io import BytesIO
from socket import socket, AF_UNIX, SOCK_STREAM
from sys import byteorder
from os import chmod

from PIL.JpegImagePlugin import JpegImageFile
from pipelines.models import TextToImageRequest

from pipeline import load_pipeline, infer

SOCKET = "/sandbox/inferences.sock"


def main():
    print(f"Loading pipeline")
    pipeline = load_pipeline()

    print(f"Pipeline loaded")

    with socket(AF_UNIX, SOCK_STREAM) as inference_socket:
        inference_socket.bind(SOCKET)

        chmod(SOCKET, 0o777)

        inference_socket.listen(1)

        print(f"Awaiting connections")

        connection, _ = inference_socket.accept()

        print(f"Connected")

        with connection:
            connection.setblocking(True)

            while True:
                size = int.from_bytes(connection.recv(2), byteorder)

                print(f"Awaiting message of size {size}")

                message = connection.recv(size).decode("utf-8")

                if not message:
                    print(f"Empty message received")
                    continue

                request = TextToImageRequest.model_validate_json(message)

                image = infer(request, pipeline)

                data = BytesIO()
                image.save(data, format=JpegImageFile.format)

                packet = data.getvalue()

                connection.send(len(packet).to_bytes(4, byteorder))
                connection.send(packet)


if __name__ == '__main__':
    main()
