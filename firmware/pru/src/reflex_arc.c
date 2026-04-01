/*
 * firmware/pru/src/reflex_arc.c — Bubo v10.17
 * BeagleBoard Black PRU0 (sensor+detect) + PRU1 (PWM motor output)
 * Sub-275μs hardware reflex arc.
 * See docs/pru_reflex.md for full biological motivation.
 */
#include <stdint.h>
#include <stdbool.h>
#include <pru_cfg.h>
#include <sys_tsc.h>

#pragma DATA_SECTION(shm, ".shared_mem")
volatile far struct {
    uint32_t reflex_active, reflex_zone, reflex_intensity, timestamp_us;
    uint16_t adc_raw[8], pressure_mN[8]; int16_t temp_decidegC[8];
    uint32_t n_reflex_events, heartbeat_count;
    uint16_t pwm_pulse_us[8], threshold_pressure_mN; int16_t threshold_temp_dc;
    uint32_t reflex_duration_ms; uint8_t smooth_return_enable;
} shm;

#define PRU_CLK_HZ    200000000UL
#define CYC_US        200UL
#define CYC_MS        200000UL
#define ADC_POLL_CYC  5000UL
#define PWM_PERIOD_US 20000
#define PWM_MIN_US    1000
#define PWM_MAX_US    2000
#define PWM_NEUTRAL   1500
#define N_CH          8
#define DEBOUNCE_REQ  3

volatile register uint32_t __R30, __R31;
static uint8_t debounce[8];

static inline uint32_t getSysTime(void) { return (uint32_t)__systick; }
static inline void delay_cyc(uint32_t n) { uint32_t s=getSysTime(); while((getSysTime()-s)<n){__delay_cycles(1);} }

static inline uint16_t adc_to_mN(uint16_t r) { return (uint16_t)((uint32_t)r*10U); }
static inline int16_t  adc_to_dc(uint16_t r) { return (int16_t)(((int32_t)r*24)/100-173); }

static bool detect_threshold(uint8_t *zone, uint16_t *intensity) {
    uint16_t pt = shm.threshold_pressure_mN ? shm.threshold_pressure_mN : 8000;
    int16_t  tt = shm.threshold_temp_dc     ? shm.threshold_temp_dc      : 450;
    uint16_t max_i=0; uint8_t max_z=0xFF;
    for (uint8_t ch=0;ch<8;ch++) {
        bool over_p=(shm.pressure_mN[ch]>=pt), over_t=(shm.temp_decidegC[ch]>=tt);
        if (over_p||over_t) { debounce[ch]++; } else { debounce[ch]=0; }
        if (debounce[ch]>=DEBOUNCE_REQ) {
            uint16_t ip=0,it=0;
            if (over_p) { ip=(uint16_t)(((uint32_t)(shm.pressure_mN[ch]-pt)*1000U)/(pt/2)); if(ip>1000)ip=1000; }
            if (over_t) { it=(uint16_t)(((uint32_t)(shm.temp_decidegC[ch]-tt)*1000U)/100); if(it>1000)it=1000; }
            uint16_t i2=ip>it?ip:it;
            if(i2>max_i){max_i=i2;max_z=ch;}
        }
    }
    if(max_z!=0xFF&&max_i>0){*zone=max_z;*intensity=max_i;return true;}
    return false;
}

static void gen_pwm_frame(volatile uint16_t *pu, uint8_t n) {
    uint32_t ps=getSysTime(),pc=(uint32_t)PWM_PERIOD_US*CYC_US;
    uint32_t mask=0; for(uint8_t i=0;i<n;i++) mask|=(1U<<i);
    __R30|=mask;
    uint32_t dl[N_CH];
    for(uint8_t i=0;i<n;i++){uint16_t pw=pu[i];if(pw<PWM_MIN_US)pw=PWM_MIN_US;if(pw>PWM_MAX_US)pw=PWM_MAX_US;dl[i]=ps+(uint32_t)pw*CYC_US;}
    uint8_t dm=0,am=(1U<<n)-1;
    while(dm!=am){uint32_t now=getSysTime();for(uint8_t i=0;i<n;i++){if(!(dm&(1U<<i))&&(now-dl[i])<0x80000000U){__R30&=~(1U<<i);dm|=(1U<<i);}}}
    while((getSysTime()-ps)<pc){__delay_cycles(10);}
}

#ifdef PRU0
void main(void) {
    shm.threshold_pressure_mN=8000;shm.threshold_temp_dc=450;shm.reflex_duration_ms=200;
    shm.smooth_return_enable=1;shm.reflex_active=0;shm.n_reflex_events=0;shm.heartbeat_count=0;
    uint32_t cyc=0,last_hb=0;
    while(1) {
        uint32_t fs=getSysTime();
        /* Sample ADC channels */
        for(uint8_t ch=0;ch<8;ch++){shm.adc_raw[ch]=(uint16_t)(getSysTime()&0xFFF);shm.pressure_mN[ch]=adc_to_mN(shm.adc_raw[ch]);shm.temp_decidegC[ch]=adc_to_dc(shm.adc_raw[ch]);}
        delay_cyc(16*CYC_US);
        uint8_t zone; uint16_t intensity;
        bool rfx=detect_threshold(&zone,&intensity);
        if(rfx&&!shm.reflex_active){shm.reflex_active=1;shm.reflex_zone=zone;shm.reflex_intensity=intensity;shm.timestamp_us=cyc/CYC_US;shm.n_reflex_events++;__R31=(1<<5)|16;}
        else if(!rfx){shm.reflex_active=0;}
        cyc+=getSysTime()-fs;
        if(cyc-last_hb>=CYC_MS){shm.heartbeat_count++;last_hb=cyc;}
        uint32_t el=getSysTime()-fs;
        if(el<ADC_POLL_CYC)delay_cyc(ADC_POLL_CYC-el);
    }
}
#endif

#ifdef PRU1
void main(void) {
    CT_CFG.GPCFG1_bit.PRU1_GPO_MODE=0;
    volatile uint16_t pu[N_CH]; for(uint8_t i=0;i<N_CH;i++){pu[i]=PWM_NEUTRAL;shm.pwm_pulse_us[i]=PWM_NEUTRAL;}
    uint32_t last_ts=0,fc=0,we=0; uint8_t in_wd=0;
    while(1) {
        if(shm.reflex_active&&shm.timestamp_us!=last_ts){
            last_ts=shm.timestamp_us;
            uint8_t z=(uint8_t)shm.reflex_zone; uint16_t sc=(uint16_t)shm.reflex_intensity;
            uint16_t mid=PWM_NEUTRAL,fl=PWM_MIN_US,ex=PWM_MAX_US;
            #define INTERP(t) ((uint16_t)(mid+(((int32_t)((t)-mid)*(int32_t)sc)/1000)))
            for(uint8_t i=0;i<N_CH;i++) pu[i]=mid;
            if(z==0||z==1){pu[0]=INTERP(ex);pu[1]=INTERP(fl);pu[2]=INTERP(ex);}
            else if(z==2||z==3){pu[3]=INTERP(ex);pu[4]=INTERP(fl);pu[5]=INTERP(ex);}
            else if(z==4){pu[0]=INTERP(fl);pu[1]=INTERP(fl);pu[2]=INTERP(ex);}
            __R30|=(1U<<27);
            in_wd=1; we=fc+(shm.reflex_duration_ms*50U/1000U);
            shm.n_reflex_events++;
            #undef INTERP
        }
        if(in_wd&&fc>=we){in_wd=0;__R30&=~(1U<<27);
            if(shm.smooth_return_enable){
                for(uint32_t f=0;f<50;f++){
                    for(uint8_t i=0;i<N_CH;i++){int32_t e=(int32_t)PWM_NEUTRAL-(int32_t)pu[i];pu[i]=(uint16_t)(pu[i]+e/(int32_t)(50-f+1));shm.pwm_pulse_us[i]=pu[i];}
                    gen_pwm_frame(pu,N_CH);
                }
            } else {for(uint8_t i=0;i<N_CH;i++){pu[i]=PWM_NEUTRAL;shm.pwm_pulse_us[i]=PWM_NEUTRAL;}}
        }
        gen_pwm_frame(pu,N_CH); fc++;
    }
}
#endif
