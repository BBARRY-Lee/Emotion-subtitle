from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .models import User
import classification_model.prediction as pred

def model_view(request):
    if request.method == "POST":
        subtitle = request.user.subtitle.url
        user = request.user
        print(subtitle)
        print(user)
        model_path = '/home/ubuntu/model/pre-trained/230206_kobert_epoch35.pth'
        pred.eval(input_path = ("/home/ubuntu/myapp/"+subtitle), model_path = model_path)
    return render(request, "users/model.html")

def index_view(request):
    if request.method == "GET":
        redirect("user:login")
    return render(request, "users/login.html")


def login_view(request):

    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        print(username)
        print(password)
        user = authenticate(username=username, password=password)
        if user is not None:
            print("인증성공")
            login(request, user)
            return redirect("user:upload")
        else:
            print("인증실패")
    return render(request, "users/login.html")


def logout_view(request):
    logout(request)
    return redirect("user:login")


def signup_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        email = request.POST["email"]
        user = User.objects.create_user(username, email, password)
        user.save()
        return redirect("user:login")
    return render(request, "users/signup.html")


def sign_view(request):
    if request.method == "POST":
        print(request.POST)


def check_view(request):
    if request.method == "POST":
        print(request.POST)
        print(request.user.profile_img.url)

        return redirect("user:login")
    return render(request, "users/login.html")


def upload_view(request):
    if request.method == "POST":
        print(request.FILES)
        video = request.FILES["video"]
        subtitle = request.FILES["subtitle"]
        print(subtitle)
        user = request.user
        user.video = video
        user.subtitle = subtitle
        user.save()
        return redirect("user:model")
    return render(request, "users/upload.html")
