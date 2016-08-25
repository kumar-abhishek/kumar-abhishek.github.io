---
layout: post
title: "Extending ssh logged in sessions for remote servers"
date: 2009-08-23
---
<div dir="ltr" style="text-align: left;" trbidi="on">Many a times, after sshing to a remote server, the connection automatically dies after a few minutes of inactivity. These are few workarounds to overcome this problem:  
* open /etc/ssh/sshd_config file as root and add the line:  
ClientAliveInterval 60  
This keeps the client connection to remote server alive for 60 secs. You can increase this. Restart your ssh server after making this change.  

*On your local computer, edit /etc/ssh/ssh_config and add the line:  
ServerAliveInterval 60</div>
