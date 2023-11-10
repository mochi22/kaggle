def make_file(image_id,size):
    file_string = """
package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
    "github.com/nfnt/resize"
)

// cropAndSave crops the image based on provided coordinates and saves to the specified folder.
func cropAndSave(img image.Image, x, y, d int, filename, dirOut string) {
	rect := image.Rect(x, y, x+d, y+d)
	subImg := img.(interface {
		SubImage(r image.Rectangle) image.Image
	}).SubImage(rect)
    resizedImg := resize.Resize(SIZE, SIZE, subImg, resize.Lanczos3)
	
	outPath := filepath.Join(dirOut, fmt.Sprintf("%s_%d_%d%s", strings.TrimSuffix(filename, filepath.Ext(filename)), y, x, filepath.Ext(filename)))
	outFile, err := os.Create(outPath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer outFile.Close()

	if filepath.Ext(filename) == ".jpg" || filepath.Ext(filename) == ".jpeg" {
		jpeg.Encode(outFile, resizedImg, nil)
	} else if filepath.Ext(filename) == ".png" {
		png.Encode(outFile, resizedImg)
	}
}

func main() {
	filename := "IMAGE_ID.png" // Change this to your image file here
	dirIn := "/kaggle/input/UBC-OCEAN/_images/"
	dirOut := "/data/IMAGE_ID/"
	d := 1024  // desired width of each output image

	imgFile, err := os.Open(filepath.Join(dirIn, filename))
	if err != nil {
		fmt.Println("Error opening image:", err)
		return
	}
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Error decoding image:", err)
		return
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var wg sync.WaitGroup
    
	for y := 0; y <= height-d; y += d {
		for x := 0; x <= width-d; x += d {
			wg.Add(1)
			go func(x, y int) {
				defer wg.Done()
				cropAndSave(img, x, y, d, filename, dirOut)
			}(x, y)
		}
	}

	wg.Wait()
}

    """
    file_string = (file_string.replace("IMAGE_ID", str(image_id)).replace("SIZE", str(size)))
    with open(f"tiler_{image_id}.go", 'w') as f:
        f.write(file_string)
    file_string = """
package main

import (
	"fmt"
	"image"

	// "image/color"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

// Checks if the image is almost black
func isAlmostBlack(img image.Image) bool {
	threshold := 3 // Adjust this threshold as needed
	totalPixels := 0
	darkPixels := 0

	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Check if the pixel is dark
			if int(r) < int(threshold) && int(g) < int(threshold) && int(b) < int(threshold) {
				darkPixels++
			}
			totalPixels++
		}
	}
	// Adjust this ratio as needed. For now, it considers the image as "almost black" if 90% of the pixels are dark.
	return float64(darkPixels)/float64(totalPixels) > 0.99
}

// Processes the image
func processImage(filename string) {
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return
	}

	if isAlmostBlack(img) {
		err := os.Remove(filename)
        if err!= nil {
            return
        }
	}
}

func main() {
	dir := "/data/IMAGE_ID/" // Change this to your directory

	files, err := ioutil.ReadDir(dir)
	if err != nil {
		fmt.Println("Error reading directory:", err)
		return
	}

	var wg sync.WaitGroup
	for _, f := range files {
		if f.IsDir() {
			continue
		}

		ext := filepath.Ext(f.Name())
		if ext == ".jpg" || ext == ".png" { // You can add more extensions if needed
			wg.Add(1)
			go func(filename string) {
				defer wg.Done()
				processImage(filename)
			}(filepath.Join(dir, f.Name()))
		}
	}
	wg.Wait()
}
    """
    file_string = (file_string.replace("IMAGE_ID", str(image_id)))
    with open(f"remover_{image_id}.go", 'w') as f:
        f.write(file_string)
    return (f"tiler_{image_id}.go", f"remover_{image_id}.go")