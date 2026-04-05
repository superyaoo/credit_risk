import smtplib
import email.mime.multipart
import email.mime.text
import base64
import os
class MailSend():
    def __init__(self,user,password):
        self.__user=user
        self.__password=password
    @property
    def user(self):
        return self.__user
    @user.setter
    def user(self,user):
        self.__user=user
    @property
    def password(self):
        return self.__password
    @password.setter
    def password(self,password):
        self.__password=password   
    def getFileType(self,file):
        att = email.mime.text.MIMEText(open(file, 'rb').read(), 'base64', 'UTF-8') 
        att["Content-Type"] = 'application/octet-stream' 
        att.add_header('Content-Disposition', 'attachment', filename='=?utf-8?b?' + str(base64.b64encode(os.path.basename(file).encode('UTF-8')))[2:-1] + '?=')
        return att
    def getattachmentlist(self,attachment):
        attachmentlist=[]
        if attachment is not None:
            if isinstance(attachment,list):
                for file in attachment:
                    attachmentlist.append(self.getFileType(file))
            elif isinstance(attachment,str):
                 attachmentlist.append(self.getFileType(attachment))
        return attachmentlist
    def convertStrList(self,receivers):
        if receivers is not None:
            if isinstance(receivers,list):
                receiverStr=receivers[0]
                for receiver in receivers[1:]:
                    receiverStr=receiverStr+','+receiver
                return receiverStr
            elif isinstance(receivers,str):
                return receivers.split(',')
        return None

    def sendmail(self,subject,to,body,attachment,subtype='plain'):
        """
        subject 邮件头部
        to   邮件接收人list 列表
        body 邮件主体
        attachment 附件列表
        """
        msg=email.mime.multipart.MIMEMultipart()
        msg['from']=self.__user
        msg['subject']=subject
        bodytxt =email.mime.text.MIMEText(body,subtype,'utf-8')
        msg.attach(bodytxt)
        attachmentlist=self.getattachmentlist(attachment)
        if(len(attachmentlist)>0):
            for att in attachmentlist:
                msg.attach(att)
        smtp=smtplib.SMTP_SSL('smtp.exmail.qq.com',465)
        smtp.login(user=self.__user,password=self.__password)
        msg['to']=";".join(to)
        smtp.sendmail(self.__user,to,msg.as_string())
        smtp.quit()