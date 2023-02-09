from django.urls import path
from . import views
app_name = "user"
urlpatterns = [
   
    path("", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("signup", views.signup_view, name="signup"),
    path("sign", views.sign_view, name="sign"),
    path("check", views.check_view, name="check"),
    path("upload", views.upload_view, name="upload"),
    path("model", views.model_view, name="model")
]
