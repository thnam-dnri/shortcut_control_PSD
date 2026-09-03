// Bounded Co-60 direct-preamp DAQ for the NKFADC500 at the realtime target.
//
// This local implementation keeps the vendor packet decoder and register API
// but writes the target event shape directly: 4500 samples at 250 MSPS,
// corresponding to 6 us pre-trigger plus 12 us post-trigger.

#include <TFile.h>
#include <TNamed.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include "/mnt/Data/FADC500/DAQ/lib/usb3comroot.h"
#include "/mnt/Data/FADC500/DAQ/lib/NoticeNKFADC500ROOT.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstddef>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr int SID = 1;
constexpr unsigned long RECORD_LENGTH = 128;
constexpr unsigned long RAW_EVENT_WORDS = RECORD_LENGTH * 128;
constexpr std::size_t RAW_EVENT_BYTES = RAW_EVENT_WORDS * 4;
constexpr int STORED_SAMPLES = 4500;
constexpr unsigned long SAMPLE_RATE_DIVIDER = 2;
constexpr unsigned long SAMPLE_PERIOD_NS = 4;
constexpr unsigned long COINCIDENCE_WIDTH_NS = 1000;
constexpr unsigned long DELAY_NS = 5000;
constexpr unsigned long PRETRIGGER_NS = DELAY_NS + COINCIDENCE_WIDTH_NS;
constexpr unsigned long STORED_WINDOW_NS = STORED_SAMPLES * SAMPLE_PERIOD_NS;
constexpr unsigned long POSTTRIGGER_NS = STORED_WINDOW_NS - PRETRIGGER_NS;
constexpr unsigned long BUFFER_LIMIT_KB = 40 * 1024;
constexpr int MAX_EVENTS_PER_ROOT_FILE = 100000;

struct HPGePulse250 {
    UInt_t event_id;
    Float_t waveform[STORED_SAMPLES];
    Float_t trigger_time_s;
};

unsigned long read_trigger_time_ns(const std::vector<char> &data, std::size_t event_offset)
{
    const unsigned long coarse =
        static_cast<unsigned long>(static_cast<unsigned char>(data[event_offset + 12 * 4])) |
        (static_cast<unsigned long>(static_cast<unsigned char>(data[event_offset + 13 * 4])) << 8U) |
        (static_cast<unsigned long>(static_cast<unsigned char>(data[event_offset + 14 * 4])) << 16U);
    return coarse * 1000UL;
}

bool kill_requested()
{
    FILE *handle = std::fopen("KILLME", "r");
    if (handle == nullptr) {
        return false;
    }
    std::fclose(handle);
    gSystem->Unlink("KILLME");
    return true;
}

}  // namespace

int run_co60_preamp_250msps(unsigned long threshold = 10,
                            int events = MAX_EVENTS_PER_ROOT_FILE,
                            int timeout_seconds = 300,
                            const char *run_id = "manual",
                            const char *source_tag = "co60")
{
    if (events <= 0 || events > MAX_EVENTS_PER_ROOT_FILE ||
        timeout_seconds <= 0 || threshold == 0) {
        std::fprintf(stderr,
                     "Invalid DAQ arguments: threshold=%lu events=%d (max=%d) timeout=%d\n",
                     threshold, events, MAX_EVENTS_PER_ROOT_FILE, timeout_seconds);
        return 2;
    }

    const char *output_dir = "/mnt/Data/ML_DetA/raw_data";
    const std::string source_tag_name = source_tag != nullptr ? source_tag : "unknown";
    gSystem->mkdir(output_dir, true);
    const std::string output_prefix =
        std::string(Form("%s/%s_preamp_250msps_%s_thr%lu_",
                         output_dir, source_tag_name.c_str(), run_id, threshold));
    const std::string output_path = output_prefix + "1.root";

    constexpr unsigned long threshold_channel_2 = 4095;
    constexpr unsigned long offset_channel_1 = 3100;
    constexpr unsigned long offset_channel_2 = 3845;
    constexpr unsigned long pulse_width_threshold = 120;
    constexpr unsigned long deadtime_ns = 500000;
    constexpr unsigned long trigger_lookup_table = 0xFFFE;
    constexpr unsigned long pulse_sum_width_ns = 2;
    constexpr unsigned long pulse_count_threshold = 1;
    constexpr unsigned long pulse_count_interval_ns = 1000;
    constexpr unsigned long trigger_polarity = 0;
    constexpr unsigned long adc_mode = 0;
    constexpr unsigned long pulse_width_trigger = 1;
    constexpr unsigned long self_trigger = 1;
    constexpr unsigned long pedestal_trigger = 0;
    constexpr unsigned long software_trigger = 0;
    constexpr unsigned long trigger_enable = (software_trigger << 2) |
                                              (pedestal_trigger << 1) |
                                              self_trigger;
    constexpr unsigned long trigger_mode = (pulse_width_trigger << 1);

    std::printf("DAQ_CONFIG source=%s input=direct_preamp polarity=negative ",
                source_tag_name.c_str());
    std::printf("threshold_adc=%lu sample_rate_msps=250 sample_period_ns=%lu ",
                threshold, SAMPLE_PERIOD_NS);
    std::printf("stored_samples=%d stored_window_us=%lu pretrigger_us=%lu posttrigger_us=%lu ",
                STORED_SAMPLES, STORED_WINDOW_NS / 1000UL,
                PRETRIGGER_NS / 1000UL, POSTTRIGGER_NS / 1000UL);
    std::printf("events=%d timeout_seconds=%d output_root=%s\n",
                events, timeout_seconds, output_path.c_str());

    gSystem->Load("libusb3comroot.so");
    gSystem->Load("libNoticeNKFADC500ROOT.so");

    TFile *output_file = TFile::Open(output_path.c_str(), "RECREATE");
    if (output_file == nullptr || output_file->IsZombie()) {
        std::fprintf(stderr, "Cannot create ROOT output: %s\n", output_path.c_str());
        delete output_file;
        return 3;
    }

    HPGePulse250 event{};
    TTree *tree = new TTree("HPGE", "HPGe NKFADC500 waveform file at 250 MSPS");
    tree->Branch("event", &event,
                 "event_id/i:waveform[4500]/F:trigger_time_s/F");

    usb3comroot *usb = new usb3comroot;
    NKNKFADC500 *digitizer = new NKNKFADC500;
    usb->USB3Init(0);

    std::cout << "Connecting to NKFADC500" << std::endl;
    digitizer->NKFADC500open(SID, 0);
    digitizer->NKFADC500resetTIMER(SID);
    digitizer->NKFADC500reset(SID);
    digitizer->NKFADC500_ADCALIGN_500(SID);
    digitizer->NKFADC500write_DRAMON(SID, 1);
    digitizer->NKFADC500write_DRAMON(SID, 2);
    digitizer->NKFADC500_ADCALIGN_DRAM(SID);

    digitizer->NKFADC500write_PTRIG(SID, 0);
    digitizer->NKFADC500write_RL(SID, RECORD_LENGTH);
    digitizer->NKFADC500write_DSR(SID, SAMPLE_RATE_DIVIDER);
    digitizer->NKFADC500write_TLT(SID, trigger_lookup_table);
    digitizer->NKFADC500write_TRIGENABLE(SID, trigger_enable);

    digitizer->NKFADC500write_DLY(SID, 1, PRETRIGGER_NS);
    digitizer->NKFADC500write_CW(SID, 1, COINCIDENCE_WIDTH_NS);
    digitizer->NKFADC500write_DACOFF(SID, 1, offset_channel_1);
    digitizer->NKFADC500measure_PED(SID, 1);
    digitizer->NKFADC500write_THR(SID, 1, threshold);
    digitizer->NKFADC500write_POL(SID, 1, trigger_polarity);
    digitizer->NKFADC500write_PSW(SID, 1, pulse_sum_width_ns);
    digitizer->NKFADC500write_AMODE(SID, 1, adc_mode);
    digitizer->NKFADC500write_PCT(SID, 1, pulse_count_threshold);
    digitizer->NKFADC500write_PCI(SID, 1, pulse_count_interval_ns);
    digitizer->NKFADC500write_PWT(SID, 1, pulse_width_threshold);
    digitizer->NKFADC500write_DT(SID, 1, deadtime_ns);
    digitizer->NKFADC500write_TM(SID, 1, trigger_mode);

    // Keep channel 2 disabled for this single-channel HPGe run.
    digitizer->NKFADC500write_DLY(SID, 2, PRETRIGGER_NS);
    digitizer->NKFADC500write_CW(SID, 2, COINCIDENCE_WIDTH_NS);
    digitizer->NKFADC500write_DACOFF(SID, 2, offset_channel_2);
    digitizer->NKFADC500measure_PED(SID, 2);
    digitizer->NKFADC500write_THR(SID, 2, threshold_channel_2);
    digitizer->NKFADC500write_POL(SID, 2, trigger_polarity);
    digitizer->NKFADC500write_PSW(SID, 2, pulse_sum_width_ns);
    digitizer->NKFADC500write_AMODE(SID, 2, adc_mode);
    digitizer->NKFADC500write_PCT(SID, 2, pulse_count_threshold);
    digitizer->NKFADC500write_PCI(SID, 2, pulse_count_interval_ns);
    digitizer->NKFADC500write_PWT(SID, 2, pulse_width_threshold);
    digitizer->NKFADC500write_DT(SID, 2, deadtime_ns);
    digitizer->NKFADC500write_TM(SID, 2, trigger_mode);

    digitizer->NKFADC500reset(SID);
    std::printf("DAQ_READBACK rl=%lu sr=%lu trig_enable=0x%lx ",
                digitizer->NKFADC500read_RL(SID),
                digitizer->NKFADC500read_DSR(SID),
                digitizer->NKFADC500read_TRIGENABLE(SID));
    std::printf("ch1_dly_ns=%lu ch1_cw_ns=%lu ch1_thr=%lu ch1_pol=%lu\n",
                digitizer->NKFADC500read_DLY(SID, 1),
                digitizer->NKFADC500read_CW(SID, 1),
                digitizer->NKFADC500read_THR(SID, 1),
                digitizer->NKFADC500read_POL(SID, 1));
    digitizer->NKFADC500start(SID);
    std::cout << "NKFADC500 Connected; acquisition started" << std::endl;

    const std::time_t start_wall = std::time(nullptr);
    std::size_t events_recorded = 0;
    std::size_t malformed_events = 0;
    unsigned long previous_trigger_ns = 0;
    unsigned long trigger_overflows = 0;
    double first_trigger_s = 0.0;
    bool have_first_trigger = false;
    bool running = true;

    while (running && events_recorded < static_cast<std::size_t>(events)) {
        unsigned long buffer_kb = digitizer->NKFADC500read_BCOUNT(SID);
        if (std::difftime(std::time(nullptr), start_wall) >= timeout_seconds) {
            std::printf("Timeout reached (%d sec). Stopping acquisition.\n", timeout_seconds);
            break;
        }
        if (kill_requested()) {
            std::printf("KILLME requested. Stopping acquisition.\n");
            break;
        }
        if (buffer_kb == 0) {
            gSystem->Sleep(1);
            continue;
        }
        if (buffer_kb > BUFFER_LIMIT_KB) {
            buffer_kb = BUFFER_LIMIT_KB;
        }

        std::size_t bytes_to_read = static_cast<std::size_t>(buffer_kb) * 1024U;
        bytes_to_read -= bytes_to_read % RAW_EVENT_BYTES;
        if (bytes_to_read == 0) {
            gSystem->Sleep(1);
            continue;
        }
        const unsigned long read_kb = static_cast<unsigned long>(bytes_to_read / 1024U);
        std::vector<char> raw_data;
        try {
            raw_data.resize(bytes_to_read);
        } catch (const std::bad_alloc &error) {
            std::fprintf(stderr, "Cannot allocate %.1f MB for digitizer buffer: %s\n",
                         bytes_to_read / 1048576.0, error.what());
            running = false;
            break;
        }
        digitizer->NKFADC500read_DATA(SID, read_kb, raw_data.data());

        std::size_t event_offset = 0;
        while (event_offset + 16 <= bytes_to_read &&
               events_recorded < static_cast<std::size_t>(events)) {
            unsigned long event_words = 0;
            event_words |= static_cast<unsigned long>(static_cast<unsigned char>(raw_data[event_offset]));
            event_words |= static_cast<unsigned long>(static_cast<unsigned char>(raw_data[event_offset + 4])) << 8U;
            event_words |= static_cast<unsigned long>(static_cast<unsigned char>(raw_data[event_offset + 8])) << 16U;
            event_words |= static_cast<unsigned long>(static_cast<unsigned char>(raw_data[event_offset + 12])) << 24U;
            const std::size_t event_bytes = static_cast<std::size_t>(event_words) * 4U;
            if (event_bytes < 128U || event_offset + event_bytes > bytes_to_read) {
                ++malformed_events;
                break;
            }
            if (event_words != RAW_EVENT_WORDS) {
                ++malformed_events;
                event_offset += event_bytes;
                continue;
            }

            const unsigned long trigger_ns = read_trigger_time_ns(raw_data, event_offset);
            if (trigger_ns < previous_trigger_ns) {
                ++trigger_overflows;
            }
            previous_trigger_ns = trigger_ns;
            const double trigger_s = trigger_ns / 1.0e9 +
                std::pow(2.0, 24.0) / 1.0e6 * trigger_overflows;

            event.event_id = static_cast<UInt_t>(events_recorded + 1);
            event.trigger_time_s = static_cast<Float_t>(trigger_s);
            for (int sample = 0; sample < STORED_SAMPLES; ++sample) {
                const std::size_t sample_offset = event_offset + 128U +
                                                   static_cast<std::size_t>(sample) * 8U;
                const unsigned int low = static_cast<unsigned int>(
                    static_cast<unsigned char>(raw_data[sample_offset]));
                const unsigned int high = static_cast<unsigned int>(
                    static_cast<unsigned char>(raw_data[sample_offset + 4U]));
                event.waveform[sample] = static_cast<Float_t>(low | (high << 8U));
            }
            tree->Fill();
            ++events_recorded;
            if (!have_first_trigger) {
                first_trigger_s = trigger_s;
                have_first_trigger = true;
            }
            if (events_recorded % 200 == 0 || events_recorded == static_cast<std::size_t>(events)) {
                const double elapsed = trigger_s - first_trigger_s;
                const double rate = elapsed > 0.0 ? (events_recorded - 1) / elapsed : 0.0;
                std::printf("Evt=%zu Rate=%.1f cps Buffer=%lu KB\n",
                            events_recorded, rate, buffer_kb);
            }
            event_offset += event_bytes;
        }
    }

    digitizer->NKFADC500stop(SID);
    digitizer->NKFADC500close(SID);
    usb->USB3Exit(0);

    output_file->cd();
    TNamed source("source", source_tag_name.c_str());
    TNamed sample_rate("sample_rate_msps", "250");
    TNamed sample_period("sample_period_ns", "4");
    TNamed stored_samples("stored_samples", "4500");
    TNamed pretrigger("pretrigger_us", "6");
    TNamed posttrigger("posttrigger_us", "12");
    TNamed trigger_threshold("trigger_threshold_adc", Form("%lu", threshold));
    source.Write();
    sample_rate.Write();
    sample_period.Write();
    stored_samples.Write();
    pretrigger.Write();
    posttrigger.Write();
    trigger_threshold.Write();
    tree->Write();
    output_file->Close();
    delete digitizer;
    delete usb;

    std::printf("DAQ_FINISHED events_recorded=%zu malformed_events=%zu output_root=%s\n",
                events_recorded, malformed_events, output_path.c_str());
    return events_recorded == static_cast<std::size_t>(events) ? 0 : 1;
}
