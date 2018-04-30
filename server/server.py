from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
from pythonsource.Predictions import Predictions

 
# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
       

  # GET
    def do_GET(self):

        #get the predictions source
        pred = Predictions()
    # Send response status code
        self.send_response(200)    
        # Send headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type','text/html')
        self.end_headers()

        parsed = urlparse.urlparse(self.path)
        parameters = urlparse.parse_qs(parsed.query)
        print(parameters)
        if(True):
            if(parameters["func"][0] == "getselectionimages"):
                message = pred.getselectionimages()
            elif(parameters["func"][0] == "suggestbooks"):
                bookin = []
                for b in range(50):
                    book = -1
                    try:
                        book = int(parameters["b"+str(b)][0])
                        bookin.append(book)
                    except:
                        break
                message = pred.suggestBooks(bookin)
            else:
                message = "Not function set!"
        
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
 
def run():
    print('starting server...')
    

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()
 
 
run()