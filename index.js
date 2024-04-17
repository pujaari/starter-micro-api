// Fungsi untuk menghitung IPK
function hitungIPK(nilai) {
    var totalBobot = 0;
    var totalSKS = 0;
    
    // Loop melalui setiap nilai
    for (var i = 0; i < nilai.length; i++) {
        totalBobot += nilai[i].nilai * nilai[i].sks;
        totalSKS += nilai[i].sks;
    }
    
    // Menghitung IPK
    var ipk = totalBobot / totalSKS;
    
    return ipk;
}

// Data dummy nilai mahasiswa
var nilaiMahasiswa = [
    { matkul: "Matematika", nilai: 3.5, sks: 4 },
    { matkul: "Fisika", nilai: 3.2, sks: 3 },
    { matkul: "Kimia", nilai: 3.8, sks: 3 },
    { matkul: "Biologi", nilai: 3.6, sks: 4 },
    { matkul: "Bahasa Inggris", nilai: 3.9, sks: 3 }
];

// Memanggil fungsi untuk menghitung IPK
var ipkMahasiswa = hitungIPK(nilaiMahasiswa);

// Menampilkan hasil IPK
console.log("IPK Mahasiswa: " + ipkMahasiswa.toFixed(2));
