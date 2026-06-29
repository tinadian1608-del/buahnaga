from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get-status-api/', views.get_status_api, name='get_status_api'),
    # Pastikan 2 baris url name di bawah ini sudah terdaftar:
    path('simpan-riwayat/', views.simpan_riwayat, name='simpan_riwayat'),
    path('riwayat/', views.riwayat_deteksi, name='riwayat_deteksi'),
]