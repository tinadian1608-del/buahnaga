from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
from .models import RiwayatDeteksi

import os
import cv2
import numpy as np



# ==============================================================================
# KONFIGURASI MODEL
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'detektor', 'model', 'best.pt')

try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

last_status = "Menunggu Objek..."
last_deskripsi = "Posisikan buah naga di depan kamera."


# ==============================================================================
# ANALISIS WARNA
# ==============================================================================

def analisis_kematangan_warna(crop_img):
    if crop_img is None or crop_img.size == 0:
        return "Tidak Diketahui", "Area buah tidak valid."

    avg_bgr = np.mean(crop_img, axis=(0, 1))
    b_avg, g_avg, r_avg = avg_bgr

    if r_avg > g_avg * 1.25:
        return "Buah Naga Matang", "Pigmentasi merah dominan."
    elif g_avg > r_avg * 0.95:
        return "Buah Naga Mentah", "Warna hijau masih dominan."
    else:
        return "Buah Naga Setengah Matang", "Terjadi transisi warna merah-hijau."


# ==============================================================================
# PARSING HASIL YOLO
# ==============================================================================

def parsing_hasil_model(nama_kelas, potongan_buah, confidence):
    nama_kelas = nama_kelas.lower().strip().replace('_', ' ')

    if 'setengah' in nama_kelas or 'half' in nama_kelas:
        return (
            "Buah Naga Setengah Matang",
            f"YOLOv8 mendeteksi setengah matang ({confidence:.2f}%)."
        )

    elif 'mentah' in nama_kelas:
        return (
            "Buah Naga Mentah",
            f"YOLOv8 mendeteksi mentah ({confidence:.2f}%)."
        )

    elif 'matang' in nama_kelas or 'ripe' in nama_kelas:
        return (
            "Buah Naga Matang",
            f"YOLOv8 mendeteksi matang ({confidence:.2f}%)."
        )

    elif 'buah naga' in nama_kelas:
        return analisis_kematangan_warna(potongan_buah)

    return (
        "Bukan Buah Naga",
        "Objek tidak sesuai karakteristik buah naga."
    )


# ==============================================================================
# STREAMING KAMERA
# ==============================================================================

def gen_frames():
    global last_status, last_deskripsi

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()

        if not success:
            break

        if model is not None:

            results = model(frame, conf=0.40, verbose=False)
            result = results[0]

            if len(result.boxes) > 0:

                box = result.boxes[0]

                confidence = float(box.conf[0]) * 100
                class_id = int(box.cls[0])

                nama_kelas = result.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                potongan_buah = frame[y1:y2, x1:x2]

                last_status, last_deskripsi = parsing_hasil_model(
                    nama_kelas,
                    potongan_buah,
                    confidence
                )

                frame = result.plot()

            else:
                last_status = "Bukan Buah Naga"
                last_deskripsi = "Objek tidak dikenali."

        ret, buffer = cv2.imencode('.jpg', frame)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )

    camera.release()


def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def get_status_api(request):
    return JsonResponse({
        'status': last_status,
        'deskripsi': last_deskripsi
    })


# ==============================================================================
# HALAMAN UTAMA
# ==============================================================================

def index(request):

    konteks = {
        'judul_sistem': 'DETEKSI REAL-TIME KEMATANGAN BUAH NAGA MENGGUNAKAN YOLOv8',
        'pake_kamera': request.GET.get('mode') == 'kamera'
    }

    if request.method == 'POST' and request.FILES.get('gambar_buah'):

        if model is None:
            konteks['status'] = "Error"
            konteks['deskripsi'] = "Model YOLO gagal dimuat."
            return render(request, 'detektor/index.html', konteks)

        gambar = request.FILES['gambar_buah']

        fs = FileSystemStorage()
        nama_file = fs.save(gambar.name, gambar)
        jalur_gambar = fs.path(nama_file)

        results = model(jalur_gambar, conf=0.40)
        result = results[0]

        if len(result.boxes) > 0:

            img_with_boxes = result.plot()

            output_dir = os.path.join(
                BASE_DIR,
                'media',
                'predicted'
            )

            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(
                output_dir,
                nama_file
            )

            cv2.imwrite(output_path, img_with_boxes)

            box = result.boxes[0]

            confidence = float(box.conf[0]) * 100
            class_id = int(box.cls[0])

            nama_kelas = result.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            potongan = result.orig_img[y1:y2, x1:x2]

            status, deskripsi = parsing_hasil_model(
                nama_kelas,
                potongan,
                confidence
            )

            RiwayatDeteksi.objects.create(
                status=status,
                deskripsi=deskripsi,
                image_path=f"/media/predicted/{nama_file}"
            )
            # ... baris kode analisis YOLO Anda di atas ...
            status, deskripsi = parsing_hasil_model(
                nama_kelas,
                potongan,
                confidence
            )

            # SIMPAN LANGSUNG KE DATA BASE DI SINI
            url_prediksi = f"/media/predicted/{nama_file}"
            RiwayatDeteksi.objects.create(
                status=status,
                deskripsi=deskripsi,
                image_path=url_prediksi  # Pastikan field ini sesuai dengan pemanggilan di riwayat.html
            )

            konteks.update({
                'url_gambar': url_prediksi,
                'status': status,
                'deskripsi': deskripsi
            })

            konteks.update({
                'url_gambar': f"/media/predicted/{nama_file}",
                'status': status,
                'deskripsi': deskripsi
            })

        else:

            konteks.update({
                'url_gambar': fs.url(nama_file),
                'status': 'Bukan Buah Naga',
                'deskripsi': 'Objek tidak terdeteksi.'
            })

    return render(
        request,
        'detektor/index.html',
        konteks
    )

# ==============================================================================
# PROSES SIMPAN RIWAYAT (TAMBAHKAN FUNGSI INI)
# ==============================================================================

def simpan_riwayat(request):
    if request.method == 'POST':
        status = request.POST.get('status')
        deskripsi = request.POST.get('deskripsi')
        url_gambar = request.POST.get('url_gambar')
        
        # Menyimpan data yang dikirim dari form ke database
        if status and deskripsi:
            RiwayatDeteksi.objects.create(
                status=status,
                deskripsi=deskripsi,
                image_path=url_gambar
            )
            
    # Setelah berhasil menyimpan, redirect kembali ke halaman utama
    from django.shortcuts import redirect
    return redirect('index')


# ==============================================================================
# HALAMAN RIWAYAT (SESUAIKAN NAMA FUNGSI)
# ==============================================================================

def riwayat_deteksi(request): # <- Ubah nama fungsi ini agar serasi dengan {% url 'riwayat_deteksi' %}
    riwayat = RiwayatDeteksi.objects.all().order_by('-id')
    return render(
        request,
        'detektor/riwayat.html',
        {
            'riwayat': riwayat
        }
    )

# ==============================================================================
# HALAMAN RIWAYAT
# ==============================================================================

def halaman_riwayat(request):

    riwayat = RiwayatDeteksi.objects.all().order_by('-id')

    return render(
        request,
        'detektor/riwayat.html',
        {
            'riwayat': riwayat
        }
    )
    
