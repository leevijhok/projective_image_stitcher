Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

* `make sync_data_to_s3` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/`.
* `make sync_data_from_s3` will use `aws s3 sync` to recursively sync files from `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/` to `data/`.


The following command-line arguments are available for demoing images:

``-folderName``
   The name of the target (demo) folder.

``-calibName``
   The name of the folder containing checkerboard calibration images.

Usage
=====

To pre-process images using the provided Python script, run the following command in your terminal:

.. code-block:: bash

   python script_name.py -folderName path_to_demo_folder -calibName path_to_calibration_images_folder

Replace ``script_name.py`` with the name of your Python script. By default, the script expects the demo folder to be located at ``data/processed/demo_1`` and the folder containing checkerboard calibration images to be located at ``data/processed/Chessboard``. If your folders are located elsewhere, you can specify their paths using the ``-folderName`` and ``-calibName`` arguments respectively.

Example
=======

For example, to pre-process images with a demo folder named ``my_demo`` located at ``/path/to/my_demo`` and calibration images located at ``/path/to/calibration_images``, you would run:

.. code-block:: bash

   python script_name.py -folderName /path/to/my_demo -calibName /path/to/calibration_images

This command will pre-process the images using the specified folders.

Note
====

If you are unfamiliar with using command-line arguments, ensure you have Python installed on your system and navigate to the directory containing the script in your terminal before executing the above command.
