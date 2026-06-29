from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from ultralytics import YOLO
from .models import RiwayatDeteksi

import os
import cv2
import numpy as np
import time



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
last_image_url = None
last_saved_status = None
last_saved_time = 0.0


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
    global last_status, last_deskripsi, last_image_url, last_saved_status, last_saved_time

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

                status, deskripsi = parsing_hasil_model(
                    nama_kelas,
                    potongan_buah,
                    confidence
                )

                last_status = status
                last_deskripsi = deskripsi

                frame = result.plot()

                # Simpan otomatis ke database jika terdeteksi buah naga dengan cooldown
                valid_statuses = ["Buah Naga Matang", "Buah Naga Setengah Matang", "Buah Naga Mentah"]
                if status in valid_statuses:
                    current_time = time.time()
                    if status != last_saved_status or (current_time - last_saved_time) > 5.0:
                        filename = f"live_{int(current_time)}.jpg"
                        output_dir = os.path.join(BASE_DIR, 'media', 'predicted')
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, filename)

                        cv2.imwrite(output_path, frame)

                        url_prediksi = f"/media/predicted/{filename}"
                        last_image_url = url_prediksi

                        try:
                            RiwayatDeteksi.objects.create(
                                status=status,
                                deskripsi=deskripsi,
                                image_path=url_prediksi
                            )
                            print(f"Berhasil menyimpan riwayat deteksi kamera: {status}")
                        except Exception as e:
                            print(f"Gagal menyimpan riwayat live cam ke database: {e}")

                        last_saved_status = status
                        last_saved_time = current_time

            else:
                last_status = "Bukan Buah Naga"
                last_deskripsi = "Objek tidak dikenali."
                # Reset status simpan agar deteksi berikutnya langsung tersimpan
                last_saved_status = None

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
        'deskripsi': last_deskripsi,
        'image_url': last_image_url
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

            url_prediksi = f"/media/predicted/{nama_file}"
            konteks.update({
                'url_gambar': url_prediksi,
                'status': status,
                'deskripsi': deskripsi
            })

        else:

            konteks.update({
                'url_gambar': fs.url(nama_file),
                'status': 'Bukan Buah Naga',
                'deskripsi': 'Objek tidak terdeteksi.'
            })

    status_filter = request.GET.get('status')
    riwayat_list = RiwayatDeteksi.objects.all().order_by('-id')

    if status_filter == 'matang':
        riwayat_list = riwayat_list.filter(status='Buah Naga Matang')
    elif status_filter == 'setengah':
        riwayat_list = riwayat_list.filter(status='Buah Naga Setengah Matang')
    elif status_filter == 'mentah':
        riwayat_list = riwayat_list.filter(status='Buah Naga Mentah')

    paginator = Paginator(riwayat_list, 10)  # Batas 10 data per halaman
    page = request.GET.get('page')
    try:
        riwayat_page = paginator.page(page)
    except PageNotAnInteger:
        riwayat_page = paginator.page(1)
    except EmptyPage:
        riwayat_page = paginator.page(paginator.num_pages)

    konteks.update({
        'riwayat': riwayat_page,
        'status_filter': status_filter
    })
    return render(
        request,
        'detektor/index.html',
        konteks
    )

# ==============================================================================
# PROSES SIMPAN RIWAYAT
# ==============================================================================

def simpan_riwayat(request):
    if request.method == 'POST':
        status = request.POST.get('status')
        deskripsi = request.POST.get('deskripsi')
        url_gambar = request.POST.get('url_gambar')
        
        if status and deskripsi:
            RiwayatDeteksi.objects.create(
                status=status,
                deskripsi=deskripsi,
                image_path=url_gambar
            )
            
    from django.shortcuts import redirect
    return redirect('/?tab=history')


# ==============================================================================
# HALAMAN RIWAYAT
# ==============================================================================

def riwayat_deteksi(request):
    from django.shortcuts import redirect
    return redirect('/?tab=history')

def halaman_riwayat(request):
    from django.shortcuts import redirect
    return redirect('/?tab=history')

    
