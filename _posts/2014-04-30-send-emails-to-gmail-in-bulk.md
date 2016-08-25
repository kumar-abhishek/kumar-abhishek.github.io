---
layout: post
title: "Sending emails to Gmail in bulk with attachments put in a directory"
date: 2009-06-07
---

<div dir="ltr" style="text-align: left;" trbidi="on">

<div style="border: 1px solid; overflow-x: scroll; padding: 1em;">  

<pre>#!/usr/bin/python
#this python code, sends emails to a particular gmail address ,
#with each email an attachment is attached which is taken from 
#the directory specified: DIRECTORY_PATH_ATTACHMENT

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders
import os,glob

gmail_user = "your_username@gmail.com"
gmail_pwd = "your_password"

DIRECTORY_PATH_ATTACHMENT='your_directory_for_attachments'

def mail(to, subject, text, attach):
   msg = MIMEMultipart()

   msg['From'] = gmail_user
   msg['To'] = to
   msg['Subject'] = subject

   msg.attach(MIMEText(text))

   part = MIMEBase('application', 'octet-stream')
   part.set_payload(open(attach, 'rb').read())
   Encoders.encode_base64(part)
   part.add_header('Content-Disposition','attachment; filename="%s"' % os.path.basename(attach))
   msg.attach(part)
   mailServer = smtplib.SMTP("smtp.gmail.com", 587)
   mailServer.ehlo()
   mailServer.starttls()
   mailServer.ehlo()
   mailServer.login(gmail_user, gmail_pwd)
   mailServer.sendmail(gmail_user, to, msg.as_string())
   # Should be mailServer.quit(), but that crashes...
   mailServer.close()

path=DIRECTORY_PATH_ATTACHMENT
for infile in glob.glob(os.path.join(path, '*.*')):
 print "sending "+infile+"\n"
 mail(gmail_user,"this is the subject of the email","this is the body of the email",infile)
</pre>

</div>

</div>
