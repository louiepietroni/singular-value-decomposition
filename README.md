# singular-value-decomposition

This project was an exploration into the uses of singular value decomposition. It explores two uses: Image compression and character recognition.


Usage:

By running showingimages.py, the non-commented image file will be loaded and will then be compressed. It is then reconstructed to different degrees, showing the relative amount of storage required.

By running digitrecognition.py, it will load an image of handwritten digits, naively split them into individual digits, then predict each one individually usign the residuals of many training digits. This program is also able to evaluate simple expressions.
