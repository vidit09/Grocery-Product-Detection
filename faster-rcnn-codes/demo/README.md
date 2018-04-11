# Faster RCNN

## Steps to use demo.

1. Download the trained weights <a href="https://drive.google.com/open?id=1MQJ7gaQfhYIN8DMczNHddxV0fN_KmFMZ">here</a>. Unzip to get ckpts.
2. Set the path to Faster RCNN installation path in config.py
3. Run the demo with img and confidence threshold. Option --box as big or small. 
	```
	python grocery_demo.py --box=big --img=/path/to/img --conf=0.5
	``` 
4. Output will be saved in same directory.


