---
layout: post
title: "Fetching emails and their attachments from gmail using python"
date: 2009-06-07
---
<div dir="ltr" style="text-align: left;" trbidi="on">

<div style="border: 1px solid; overflow-x: scroll; padding: 1em;">  

<pre>#!/usr/bin/env python
import poplib,email,os,time,sys,string,StringIO, rfc822,MySQLdb

#for gmail
PROVIDER = "pop.gmail.com"
USER="your_username"
PASSWORD = "your_password"
PATH_FOR_ATTACHMENTS="your_full_path_for_attachment" #directory for saving attachments

############################################################

def string2time (strdate):

  """ This function converts a string from a date that contains

      a string in one of the following formats:

      - %a, %d %b %Y %H:%M:%S

      - %d %b %Y %H:%M:%S

      (since some emails come with a date field in the first format,

       and others come with a date field in the second format)

  """

  ## these 3 if blocks are used to ignore the GMT/.. offset

  if not strdate.find('(') == -1:

    strdate = strdate[0:strdate.find('(')].strip()

  if not strdate.find('+') == -1:

    strdate = strdate[0:strdate.find('+')].strip()

  if not strdate.find('-') == -1:

    strdate = strdate[0:strdate.find('-')].strip()

  ## convert the date string into a  9-item tuple

  try:

    t = time.strptime(strdate, "%a, %d %b %Y %H:%M:%S")

  except ValueError, e:

    t = time.strptime(strdate, "%d %b %Y %H:%M:%S")

  return t

############################################################

def decode_field(field): #just decodes the header

  field = field.replace('\r\n','')

  field = field.replace('\n','')

  list = email.Header.decode_header(field)#This function returns a list of (decoded_string, charset)pairs containing each of the decoded parts of the header. charset is None for non-encoded parts of header, otherwise a lower case string containing the name of the character set specified in the encoded string.
  decoded = " ".join(["%s" % k for (k,v) in list])
  return decoded

def handleMsg(msg, is_subpart=False):
  filenames = []
  attachedcontents = []
  global text,attachments,fieldFrom, fieldSubject, fieldTime,fieldTo,fieldCc,fieldBcc
  while isinstance(msg.get_payload(),email.Message.Message):
 msg=msg.get_payload() #isinstance(object, classinfo):Returns true if the object argument is an instance of the classinfo argument, or of a (direct or indirect) subclass thereof.

  if not is_subpart:

    fieldFrom = ""

    fieldSubject = ""

    fieldTime = None    # fieldTime is a 9-item tuple

    text = ""           # the text contents of a message

    attachments = ""
    fieldTo=""
    fieldCc=""
    fieldBcc=""

    if fieldFrom == "" and msg['From'] != None:

  fieldTo = fieldTo+ decode_field(msg['To'])

  if msg['Cc'] != None:

   fieldCc +=decode_field(msg['Cc'])

  if msg['Bcc'] != None:

   fieldBcc += decode_field(msg['Bcc'])

  fieldFrom = decode_field(msg['From'])  
    if fieldSubject == "" and msg['Subject'] != None:

  fieldSubject = decode_field(msg['Subject'])

    if fieldTime == None and msg['Date'] != None:
  fieldTime= msg['Date']
  # print fieldTime
  fieldTime = string2time(msg['Date'])
#  print fieldTime

  fieldTime = time.strftime("%Y-%m-%d  %H:%M:%S", fieldTime)

  if msg.is_multipart(): #recursively calling for each subpart in the email. Actually email may consist of the same for submsg in msg.get_payload(): #  email in different formats, say: text/plain as well as text/html. we will handleMsg(submsg, True)
 #extract the text/plain part by recursively traversing each subpart.

   if msg.get_content_type() == 'text/plain':
           text += "\n%s" % msg.get_payload(decode=1)

#code for handling attachments starts here
  fn = msg.get_filename()
  if fn <> None:
 filenames.append(fn)
 decoded = email.base64mime.decode(msg.get_payload()) #decoding the attachment
 attachedcontents.append(decoded)

  first_filename=""

  os.chdir(PATH_FOR_ATTACHMENTS)
  for i in range(len(filenames)):
  if attachedcontents is not None:
   fp = file(filenames[i], "wb")
   first_filename=filenames[0]
   fp.write(attachedcontents[i])
   fp.close()
#         print 'Found and saved attachment "' + filenames[i] + '".'
#code for handling attachments ends here
try:
    client = poplib.POP3_SSL(PROVIDER,995) #make a connection with the server.
except:
    print "Error: Provider not found."
    sys.exit(1)

client.user(USER)
client.pass_(PASSWORD)
number_of_mails = len(client.list()[1]) #count the no. of mails

print number_of_mails

for i in range(1,number_of_mails+1) #for all mails
    lines = client.retr(i)[1]
    mailstring = string.join(lines, "\n")
    ## parse message

    msg = email.message_from_string (mailstring)

    ## handle Email.message object

    title = handleMsg( msg)
    client.quit()
    first_filename=""
    print "Time:"+fieldTime
    print "From:"+fieldFrom #prints the sender name &  email address
    print "To:"+fieldTo
    print "Cc: %s" % fieldCc
    print "Bcc: %s" % fieldBcc
    print "Subject:"+fieldSubject # prints the mail's subject
    print "Body:"
    print text  #printing the mails' body

 </pre>

</div>

</div>
