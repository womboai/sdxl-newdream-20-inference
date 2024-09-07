from io import BytesIO
from multiprocessing.connection import Listener
from os import chmod
from pathlib import Path

from PIL.JpegImagePlugin import JpegImageFile
from pipelines.models import TextToImageRequest

from pipeline import load_pipeline, infer

SOCKET = Path(__file__).parent / "inferences.sock"


def main():
    print(f"Loading pipeline")
    pipeline = load_pipeline()

    print(f"Pipeline loaded")

    with Listener(str(SOCKET)) as listener:
        chmod(SOCKET, 0o777)

        print(f"Awaiting connections")
        with listener.accept() as connection:
            print(f"Connected")

            while True:
                request = TextToImageRequest.model_validate_json(connection.recv_bytes().decode("utf-8"))

                image = infer(request, pipeline)

                data = BytesIO()
                image.save(data, format=JpegImageFile.format)

                packet = data.getvalue()

                connection.send_bytes(packet)


if __name__ == '__main__':
    main()
