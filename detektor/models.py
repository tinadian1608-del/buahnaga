from django.db import models

# Create your models here.
from django.db import models

class RiwayatDeteksi(models.Model):
    status = models.CharField(max_length=100)
    deskripsi = models.TextField()
    image_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        
