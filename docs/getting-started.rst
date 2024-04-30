Getting started
===============

NOTE: This is based on the cookie-cutter date science template. 
There are a bunch of folders with nothing in them. For example, there are no models.


Installing Environment and Running Demo with Sphinx
===================================================

Activate Environment
--------------------

If you haven't already, activate your Python environment where you want to install the dependencies from ``requirements.txt``.

.. code-block:: bash

   conda activate your_environment_name

Replace ``your_environment_name`` with the name of your Anaconda environment.

Install Dependencies
---------------------

Install the Python dependencies listed in ``requirements.txt``.

.. code-block:: bash

   pip install -r requirements.txt

Placement of Data
------------------

#. Insert the images to-be-stitched into the into data\processed into a folder called demo_1. 

#. Insert the calibration checkerboard images into data\processed into a folder called Chessboard.

#. Place the images into data\raw instead in demo_1_high and Chessboard_high, if you want images resized.

#. If any of these folders do not exist, create them. The code doesn't work without them.

#. If you wish to use calibration or demo data from other folders other than the standard folders, see commands.rst

Integrate with Sphinx
----------------------

#. Navigate to the directory where you have your Sphinx documentation project.

#. Open or create the ``getting-started.rst`` file in the ``source`` directory (or whatever you named it during Sphinx setup).

#. Add a section or a paragraph in the ``getting-started.rst`` file to explain how to run ``demo.py``. For example:

.. code-block:: rst

   Running the Demo
   ~~~~~~~~~~~~~~~~

   To run the demo, simply execute the following command in your terminal:

   .. code-block:: bash

      python -m src.demo

   This will execute the demo script and show the output.

   Adjust the path and command according to your project structure if needed.

Build the Documentation
-------------------------

Navigate to your Sphinx project directory in the terminal and build the documentation.

.. code-block:: bash

   make html

This command will generate the HTML files for your documentation.

View the Documentation
------------------------

After the build process completes successfully, you can view the generated documentation by opening the HTML files in a web browser. The main ``index.html`` file is usually located in the ``_build/html`` directory within your Sphinx project directory.

Verify Demo Integration
------------------------

Open the documentation in your web browser and navigate to the ``getting-started`` section (or whatever you named it). You should see the instructions for running the demo along with the code snippet.

Run the Demo
--------------

Following the instructions provided in the documentation, execute the ``demo.py`` script in your terminal.

.. code-block:: bash

   python -m src.demo

This will run the demo script and display the output.


