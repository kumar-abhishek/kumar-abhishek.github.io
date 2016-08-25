---
layout: post
title: "Using Apache Tika to convert a MS word or doc or pdf file to a txt file(plain text)"
date: 2009-06-07
---
<div dir="ltr" style="text-align: left;" trbidi="on">

<div style="border: 1px solid; overflow-x: scroll; padding: 1em;">  

<pre>//compiling:javac -cp  full_path_to_your_tika_jar_file SimpleTikaExample.java
//executing:java -cp  full_path_to_your_tika_jar_file:. SimpleTikaExample name_of_your_doc_file
//dont forget ":." while executing

import java.io.*;  
import org.apache.tika.metadata.*;  
import org.apache.tika.parser.*;  
import org.apache.tika.parser.pdf.*;  
import org.apache.tika.sax.*;  
import org.xml.sax.*;  
import java.util.Properties;

public class convert {  
static public void main(String args[]) {  
try {  
 File docinputfile = new File(args[0]);  
 InputStream istream = new FileInputStream(docinputfile);  
 ContentHandler textHandler = new BodyContentHandler();  
 Metadata metadata = new Metadata();  
 metadata.add(Metadata.RESOURCE_NAME_KEY, docinputfile.getName());  
 AutoDetectParser parser = new AutoDetectParser();  
 parser.parse(istream, textHandler, metadata);  
 istream.close();  
 ostream.println("Title: " + metadata.get("title"));  
 ostream.println("Author: " + metadata.get("Author"));  
 ostream.println("content: " + textHandler.toString());
       }           catch (Exception e) {  
                           e.printStackTrace();  

                   }  
           }  
   }  
    </pre>

</div>

</div>
