#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from functools import wraps
import glob

# 환경변수 설정 - 터미널
# export EMAIL_USER="your_gmail@gmail.com"
# export EMAIL_PASSWORD="your_app_password"
# export EMAIL_RECIPIENT="recipient@gmail.com"

class EmailNotifier:
    """이메일 알림 클래스 (인라인 이미지 지원)"""
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.from_email = os.getenv('EMAIL_USER')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.to_email = os.getenv('EMAIL_RECIPIENT')
        
        if not all([self.from_email, self.password, self.to_email]):
            raise ValueError("이메일 환경변수가 설정되지 않았습니다: EMAIL_USER, EMAIL_PASSWORD, EMAIL_RECIPIENT")
    
    def send_email(self, subject, message, is_html=False, inline_images=None):
        """이메일 발송 (인라인 이미지 지원)"""
        try:
            if inline_images and is_html:
                msg = MIMEMultipart('related')
            else:
                msg = MIMEMultipart()
            
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = subject
            
            if inline_images and is_html:
                msg_alternative = MIMEMultipart('alternative')
                msg.attach(msg_alternative)
                
                html_part = MIMEText(message, 'html', 'utf-8')
                msg_alternative.attach(html_part)
                
                for cid_name, image_path in inline_images.items():
                    if self._attach_inline_image(msg, image_path, cid_name):
                        print(f"✅ 인라인 이미지 첨부: {cid_name} -> {os.path.basename(image_path)}")
            else:
                msg_type = 'html' if is_html else 'plain'
                msg.attach(MIMEText(message, msg_type, 'utf-8'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.sendmail(self.from_email, self.to_email, msg.as_string())
            
            print("✅ 이메일 발송 성공")
            return True
            
        except Exception as e:
            print(f"❌ 이메일 발송 실패: {e}")
            return False
    
    def _attach_inline_image(self, msg, image_path, cid_name):
        """인라인 이미지를 이메일에 첨부"""
        try:
            if not os.path.exists(image_path):
                print(f"⚠️  이미지 파일이 존재하지 않습니다: {image_path}")
                return False
            
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext == '.png':
                img = MIMEImage(img_data, 'png')
            elif file_ext in ['.jpg', '.jpeg']:
                img = MIMEImage(img_data, 'jpeg')
            elif file_ext == '.gif':
                img = MIMEImage(img_data, 'gif')
            else:
                img = MIMEImage(img_data)
            
            img.add_header('Content-ID', f'<{cid_name}>')
            img.add_header('Content-Disposition', 'inline')
            
            msg.attach(img)
            return True
            
        except Exception as e:
            print(f"❌ 인라인 이미지 첨부 실패: {e}")
            return False
    
    def notify_success_with_images(self, task_name, start_time, end_time, result_value, image_paths):
        """
        성공 알림 (함수 반환 이미지 경로 리스트 사용)
        
        Args:
            task_name: 작업 이름
            start_time: 시작 시간
            end_time: 종료 시간
            result_value: 함수 반환값 (이미지 경로 리스트)
            image_paths: 이미지 경로 리스트
        """
        duration = end_time - start_time
        subject = f"✅ {task_name} 완료"
        
        if not image_paths or not isinstance(image_paths, (list, tuple)):
            # 이미지가 없는 경우 기본 메시지
            message = f"""
            <html>
            <body>
                <h2 style="color: green;">✅ {task_name} 완료</h2>
                <p>작업이 성공적으로 완료되었습니다.</p>
                <ul>
                    <li><strong>완료 시간:</strong> {end_time.strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><strong>소요 시간:</strong> {duration}</li>
                    <li><strong>반환값:</strong> {result_value if result_value else "없음"}</li>
                </ul>
            </body>
            </html>
            """
            self.send_email(subject, message, is_html=True)
            return
        
        # 이미지 경로를 CID로 매핑
        inline_images = {}
        image_html = ""
        
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                cid_name = f"result_image_{i+1}"
                inline_images[cid_name] = image_path
                
                image_name = os.path.basename(image_path)
                image_title = self._generate_image_title(image_name, i+1)
                
                image_html += f"""
                <div style="text-align: center; margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fafafa;">
                    <h4 style="color: #333; margin-bottom: 15px;">📊 {image_title}</h4>
                    <img src="cid:{cid_name}" alt="{image_name}" style="max-width: 700px; width: 100%; height: auto; border: 2px solid #ddd; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="font-size: 12px; color: #666; margin-top: 10px; font-style: italic;">{image_name}</p>
                </div>
                """
            else:
                print(f"⚠️  이미지 파일을 찾을 수 없습니다: {image_path}")
        
        # HTML 메시지 생성
        message = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    line-height: 1.6; 
                    margin: 0; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    max-width: 900px; 
                    margin: 0 auto; 
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #4CAF50, #45a049);
                    color: white;
                    padding: 30px; 
                    text-align: center;
                }}
                .content {{
                    padding: 30px;
                }}
                .info-table {{ 
                    width: 100%;
                    border-collapse: collapse; 
                    margin: 20px 0;
                    background-color: #fafafa;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .info-table td {{ 
                    border: 1px solid #e0e0e0; 
                    padding: 12px 15px; 
                }}
                .info-table td:first-child {{
                    background-color: #f0f0f0;
                    font-weight: bold;
                    width: 30%;
                }}
                .success {{ color: #4CAF50; font-weight: bold; }}
                .results-section {{
                    margin-top: 40px;
                }}
                .footer {{ 
                    margin-top: 30px; 
                    padding-top: 20px; 
                    border-top: 2px solid #e0e0e0; 
                    color: #666; 
                    font-size: 12px; 
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; font-size: 28px;">✅ 작업 완료!</h1>
                    <h2 style="margin: 10px 0 0 0; font-weight: normal; opacity: 0.9;">{task_name}</h2>
                </div>
                
                <div class="content">
                    <h3 style="color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">📋 작업 정보</h3>
                    <table class="info-table">
                        <tr>
                            <td>상태</td>
                            <td class="success">✅ 성공적으로 완료</td>
                        </tr>
                        <tr>
                            <td>시작 시간</td>
                            <td>{start_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>완료 시간</td>
                            <td>{end_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>총 소요 시간</td>
                            <td>{duration}</td>
                        </tr>
                        <tr>
                            <td>생성된 이미지 수</td>
                            <td><strong>{len([p for p in image_paths if os.path.exists(p)])}개</strong></td>
                        </tr>
                    </table>
                    
                    <div class="results-section">
                        <h3 style="color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">🎨 생성된 결과 이미지</h3>
                        {image_html if image_html else "<p>표시할 이미지가 없습니다.</p>"}
                    </div>
                    
                    <div class="footer">
                        <p>📧 이메일 발송 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
                        <p>이 보고서는 자동으로 생성되었습니다.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 인라인 이미지와 함께 이메일 발송
        if inline_images:
            self.send_email(subject, message, is_html=True, inline_images=inline_images)
            print(f"📧 {len(inline_images)}개 이미지가 포함된 완료 알림을 발송했습니다.")
        else:
            # 이미지가 없는 경우 간단한 메시지
            simple_message = message.replace(image_html, "<p>이미지 파일을 찾을 수 없습니다.</p>")
            self.send_email(subject, simple_message, is_html=True)
    
    def _generate_image_title(self, filename, index):
        """파일명을 기반으로 이미지 제목 생성"""
        # 파일 확장자 제거
        name_without_ext = os.path.splitext(filename)[0]
        
        # 일반적인 키워드를 한국어로 매핑
        keyword_mapping = {
            'chart': '차트',
            'graph': '그래프', 
            'plot': '플롯',
            'analysis': '분석',
            'result': '결과',
            'summary': '요약',
            'trend': '트렌드',
            'performance': '성과',
            'comparison': '비교',
            'distribution': '분포'
        }
        
        title = name_without_ext.replace('_', ' ').replace('-', ' ').title()
        
        # 키워드 매핑 적용
        for eng, kor in keyword_mapping.items():
            if eng.lower() in name_without_ext.lower():
                title = f"{kor} {title}"
                break
        else:
            title = f"결과 이미지 {index}: {title}"
        
        return title
    
    def notify_failure(self, task_name, start_time, end_time, error_msg):
        """실패 알림"""
        duration = end_time - start_time
        subject = f"❌ {task_name} 실패"
        
        message = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #fff5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background-color: #f44336; color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; }}
                .error-box {{ background-color: #ffebee; padding: 20px; margin: 20px 0; border-left: 4px solid #f44336; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                td {{ border: 1px solid #ddd; padding: 12px; }}
                td:first-child {{ background-color: #f9f9f9; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0;">❌ 작업 실패</h1>
                    <h2 style="margin: 10px 0 0 0; font-weight: normal; opacity: 0.9;">{task_name}</h2>
                </div>
                
                <div class="content">
                    <table>
                        <tr>
                            <td>상태</td>
                            <td style="color: red; font-weight: bold;">❌ 실패</td>
                        </tr>
                        <tr>
                            <td>시작 시간</td>
                            <td>{start_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>실패 시간</td>
                            <td>{end_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>소요 시간</td>
                            <td>{duration}</td>
                        </tr>
                    </table>
                    
                    <div class="error-box">
                        <h4 style="margin-top: 0; color: #d32f2f;">🚨 오류 내용</h4>
                        <pre style="white-space: pre-wrap; font-family: 'Courier New', monospace; background-color: #fff; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">{error_msg}</pre>
                    </div>
                    
                    <p style="color: #666; text-align: center; margin-top: 30px; font-size: 12px;">
                        📧 발송 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        self.send_email(subject, message, is_html=True)

def with_email_notification(task_name):
    """
    함수 실행 후 이메일 알림 데코레이터 (반환된 이미지 경로 리스트 자동 처리)
    
    Args:
        task_name: 작업 이름
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            notifier = EmailNotifier()
            start_time = datetime.now()
            
            try:
                print(f"🔄 {task_name} 시작...")
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                
                # 함수 반환값이 이미지 경로 리스트인지 확인
                if isinstance(result, (list, tuple)) and result:
                    # 이미지 파일인지 확인 (확장자 체크)
                    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
                    if all(isinstance(path, str) and 
                           os.path.splitext(path)[1].lower() in image_extensions 
                           for path in result):
                        print(f"📸 {len(result)}개의 이미지 경로를 감지했습니다.")
                        notifier.notify_success_with_images(task_name, start_time, end_time, result, result)
                        return result
                
                # 이미지 리스트가 아닌 경우 기본 알림
                notifier.notify_success_with_images(task_name, start_time, end_time, result, [])
                return result
                
            except Exception as e:
                end_time = datetime.now()
                notifier.notify_failure(task_name, start_time, end_time, str(e))
                raise
        
        return wrapper
    return decorator

