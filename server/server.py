from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
from pythonsource.Predictions import Predictions

 
# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
       

  # GET
    def do_GET(self):

        #get the predictions source
        self.pred = Predictions()

    # Send response status code
        self.send_response(200)    
        # Send headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type','text/html')
        self.end_headers()

        parsed = urlparse.urlparse(self.path)
        parameters = urlparse.parse_qs(parsed.query)
        print(parameters)
        if("func" in parameters):
            if(parameters["func"][0] == "getselectionimages"):
                message = self.pred.getselectionimages()
            elif(parameters["func"][0] == "suggestbooks"):
                bookin = []
                for b in range(50):
                    book = -1
                    try:
                        book = int(parameters["b"+str(b)][0])
                        bookin.append(book)
                    except:
                        break
                message = self.pred.suggestBooks(bookin)
            elif(parameters["func"][0] == "suggestauthorbooks"):
                bookin = []
                for b in range(50):
                    book = -1
                    try:
                        book = int(parameters["b"+str(b)][0])
                        bookin.append(book)
                    except:
                        break
                message = self.pred.suggestauthorbooks(bookin)
            elif(parameters["func"][0] == "suggestgenderbooks"):
                bookin = []
                for b in range(50):
                    book = -1
                    try:
                        book = int(parameters["b"+str(b)][0])
                        bookin.append(book)
                    except:
                        break
                message = self.pred.suggestgenderbooks(bookin)
            else:
                message = "Not function set!"
        else:
            message = "Not Allowed!"
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