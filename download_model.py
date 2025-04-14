import urllib.request

url = "http://places2.csail.mit.edu/models_places365/whole_resnet18_places365.pth.tar"
output_path = "whole_resnet18_places365.pth.tar"

print("Downloading Places365 model")
urllib.request.urlretrieve(url, output_path)
print("Download completed: ", output_path)
