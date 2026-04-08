create_project -in_memory -part xc7a100tcsg324-1
create_ip -name mig_7series -vendor xilinx.com -library ip -version 4.2 -module_name mig_test
set props [list_property [get_ips mig_test]]
foreach p $props {
    if {[string match "CONFIG.*" $p]} {
        set val [get_property $p [get_ips mig_test]]
        puts "$p = $val"
    }
}
close_project
