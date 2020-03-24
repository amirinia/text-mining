import urllib.request

response = urllib.request.urlopen('http://php.net/')

html = response.read()

print (html)