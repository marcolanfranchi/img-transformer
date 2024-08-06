## img-transformer

Turn an image into a gif with a 'glowing' effect

### Installation

1. Clone this repo and cd to root directory
    ```bash
    git clone https://github.com/marcolanfranchi/img-transformer.git
    cd img-transformer
    ```
2. Install required packages
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Add the image of your choice (png or jpg only) to the `images/` directory
2. Run the program with:
    ```bash
    python main.py IMG.png
    ```
    ... or if you want the result to be in black and white:
        ```bash
    python main.py IMG.png bw
    ```

### Example

**Input image:**

![ggg](images/IMG.jpg)

**Output gif:**

![Output gif](output/moving.gif)

