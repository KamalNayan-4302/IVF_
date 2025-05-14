import os
import logging
import openfertility as of
from openfertility.datasets import utils
import outputformat as ouf

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Dataset:
    url = "https://figshare.com/ndownloader/files/39348899"
    name = "blasto2k"
    reference = "https://doi.org/10.1038/s41597-023-02182-3"

    def __init__(self):
        # Labels
        self.target = "EXP_silver"
        self.labels_path = "Gardner_train_silver.csv"
        self.separator = ";"

        # Data
        self.data_path = "Images"
        self.data_col = "Image"

        # Set dataset path
        self.dataset_path = os.path.join(os.getcwd(), self.name)

    def download(self, force: bool = False) -> None:
        """
        Download and extract the dataset.
        If the dataset already exists, it will skip the download unless `force=True`.
        """

        if os.path.exists(self.dataset_path) and not force:
            log.info(f"{self.name} already exists at {self.dataset_path}")
            log.info("Use force=True to download again")
            return

        # Download dataset
        log.info(f"Downloading {ouf.b(self.name, return_str=True)}  {self.reference}")
        utils.download_file(self.url, f"{self.name}.zip")

        # Unzip dataset
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        log.info(f"Extracting to {self.dataset_path}")
        utils.extract_zip(f"{self.name}.zip", self.dataset_path)

        # Remove zip file
        log.info("Cleaning up...")
        os.remove(f"{self.name}.zip")
        log.info("Download and extraction complete!")


# Initialize and download dataset
blasto2k = Dataset()
blasto2k.download(force=True)
